"""Build a WorkflowConfig JSON (for `mtps run --config`) from a project directory.

A project directory holds the inputs of one system (see examples/Cu_CO2/):
  dft.json            CP2K settings and the list of input files
  init/               cp2k.inp, force_eval.inp and the initial structures coord*.xyz (with their box)
  mlp.yaml            MACE training settings (E0s, epochs, ...)   [or pass --mlp]
  tps.json, top.pdb, tps.xyz   TPS settings, topology and initial path   [tps.json: or pass --tps]
  smd/*  (optional)   smd.xyz, smd.json, plumed.dat for steered MD (see --smd-auto)

Stdlib only; writes only the output JSON (and, with --paper, patched copies of the CP2K inputs
next to it, in paper_inputs/). The config stores absolute paths, so generate it on the machine
where you run the workflow.

  python3 scripts/make_config.py PROJECT_DIR OUT.json [--mlp F] [--tps F] [--tps-xyz F]
        [--root-dir RUN_DIR] [--frames-per-split 100] [--max-gens 5] [--test-fraction 0.2]
        [--paper] [--all-init] [--foundation medium] [--smd-auto] [--init-md A.xyz B.xyz ...]

Run with --help for every option. The output directory OUT.json is written to must already exist.
"""
import argparse, ast, json, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("project"); ap.add_argument("out")
ap.add_argument("--mlp"); ap.add_argument("--tps"); ap.add_argument("--tps-xyz")
ap.add_argument("--frames-per-split", type=int, default=100)
ap.add_argument("--max-gens", type=int, default=5)
ap.add_argument("--test-fraction", type=float, default=0.2)
ap.add_argument("--root-dir", help="run dir (default: project dir)")
ap.add_argument("--foundation", nargs="*", help="MACE-MP foundation model(s) e.g. medium (no training)")
ap.add_argument("--smd-auto", action="store_true", help="run SMD with the ASE+PLUMED runner (mlp.run_md)")
ap.add_argument("--smd-stride", type=int, default=1)
ap.add_argument("--paper", action="store_true",
                help="paper settings (arXiv:2605.03737): 1000 frames/gen, 2000 MC steps, lambda=0.7 selection, 400 K smearing, Gen 0-5")
ap.add_argument("--all-init", action="store_true", help="use every init/coord*.xyz (dft.json may list fewer)")
ap.add_argument("--init-md", nargs="+", metavar="XYZ",
                help="raw initial xyz files (with box): run foundation-model MD on each first (paper SI: 400 K, 1 ns, 400 frames)")
ap.add_argument("--init-md-model", default="mace_mp:medium")
ap.add_argument("--init-md-steps", type=int, default=1_000_000)
ap.add_argument("--init-md-frames", type=int, default=400)
ap.add_argument("--init-md-auto", action="store_true", help="engine runs the MD itself (else prints per-file commands)")
a = ap.parse_args()

proj = Path(a.project).resolve()
def need(p):
    p = Path(p)
    if not p.exists():
        sys.exit(f"missing required file: {p}")
    return p

dft = json.loads(need(proj / "dft.json").read_text())
mlp_path = need(a.mlp or proj / "mlp.json")
tps = json.loads(need(a.tps or proj / "tps.json").read_text())
tps_xyz = need(a.tps_xyz or proj / "tps.xyz")

# mlp.json / train.yaml: JSON is valid YAML; E0s may be a dict or its string repr
import re
txt = mlp_path.read_text()
if mlp_path.suffix in (".yaml", ".yml"):
    m = re.search(r"^E0s:\s*(.+)$", txt, re.M)
    e0s_raw = ast.literal_eval(m.group(1).strip().strip("'\"")) if m else sys.exit("E0s not found in yaml")
else:
    e0s_raw = json.loads(txt)["E0s"]
    if isinstance(e0s_raw, str):
        e0s_raw = ast.literal_eval(e0s_raw)
e0s = {int(k): v for k, v in e0s_raw.items()}

tps_tmpl = {k: v for k, v in tps.items() if k != "mace_model_path"}
tps_tmpl["top_file"] = str(need(proj / "top.pdb")) if (proj / "top.pdb").exists() else tps_tmpl.get("top_file", "top.pdb")

if a.paper:
    tps_tmpl.update(num_select=1000, tot_mc_steps=2000, selection_scheme="paper", **{"lambda": 0.7})
    tps["tot_mc_steps"], tps["num_select"] = 2000, 1000
    a.max_gens = 6
if a.all_init:
    dft["structures"] = [f"init/{c.name}" for c in sorted((proj / "init").glob("coord*.xyz"),
                                                          key=lambda c: int(re.sub(r"\D", "", c.stem)))]
if a.paper:  # 400 K Fermi-Dirac smearing (reference run used 350 K): patched copies, originals untouched
    out_in = Path(a.out).resolve().parent / "paper_inputs"; out_in.mkdir(exist_ok=True)
    patched = []
    for f in dft["input_files"]:
        src = proj / f; txt_in = src.read_text()
        txt_in = re.sub(r"(ELECTRONIC_TEMPERATURE\s+)\d+(\.\d*)?", r"\g<1>400", txt_in)
        (out_in / src.name).write_text(txt_in); patched.append(str(out_in / src.name))
    dft["input_files"] = patched
cfg = {
    "root_dir": str(Path(a.root_dir).resolve() if a.root_dir else proj),
    "initial_structures": [str((proj / s).resolve()) for s in dft["structures"]],
    "initial_trajectory": str(tps_xyz.resolve()),
    "dft_backend": "cp2k", "mlp_backend": "mace", "tps_backend": "ops",
    "dft_settings": {
        "settings": {**dft["settings"], "filter_unconverged": True},
        "input_files": [str((proj / f).resolve()) for f in dft["input_files"]],
    },
    "mlp_settings": {"E0s": e0s, "config_file": str(mlp_path.resolve()), "valid_fraction": a.test_fraction},
    "tps_settings": {"config_template": tps_tmpl},
    "committee_seeds": [123, 234, 345],
    "n_tps_trajectories": tps.get("tot_mc_steps", 1000),
    "top_k_frames_per_generation": tps.get("num_select", 1000),
    "max_generations": a.max_gens,
    "dft_frames_per_split": a.frames_per_split,
    "cumulative_training": True,
    "selection_mode": "sampler",
    "gates": {"dft_submit": "manual", "smd_run": "manual", "tps_run": "manual", "continue_next_gen": "auto"},
}
if all((proj / f).exists() for f in ("smd.json", "smd.xyz", "plumed.dat")):
    cfg["smd_settings"] = {
        "inputs": {"smd.json": "smd.json", "smd.xyz": "coord.xyz", "plumed.dat": "plumed.dat"},
        "result": "tps.xyz",
    }
if a.init_md:
    for x in a.init_md:
        if not Path(x).exists():
            sys.exit(f"missing --init-md input: {x}")
    cfg["init_md_settings"] = {
        "inputs": [str(Path(x).resolve()) for x in a.init_md], "model": a.init_md_model,
        "temperature": 400, "time_step": 1.0, "total_steps": a.init_md_steps, "n_frames": a.init_md_frames,
        "auto_run": a.init_md_auto,
    }
    cfg["gates"]["init_md"] = "manual"
if a.foundation:
    cfg["mlp_settings"]["foundation_models"] = [f"mace_mp:{f}" if "/" not in f and not f.startswith("mace_mp:") else f
                                                for f in a.foundation]
if a.smd_auto and "smd_settings" in cfg:
    cfg["smd_settings"].update(auto_run=True, stride=a.smd_stride)
Path(a.out).write_text(json.dumps(cfg, indent=2))
print("wrote", a.out)
