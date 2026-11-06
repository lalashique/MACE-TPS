"""Check a config written by make_config.py: load it with the repo's schema, print what the engine
would do and confirm that every referenced file exists. No backend is run.

  python3 scripts/verify_config.py OUT.json
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages" / "core"))
from core.schemas import WorkflowConfig
c = WorkflowConfig.from_json(Path(sys.argv[1]))
print("inputs:", len(c.initial_structures), "files; backends:", c.dft_backend, c.mlp_backend, c.tps_backend)
print("E0s:", c.mlp_settings["E0s"]); print("cp2k input files:", c.dft_settings["input_files"])
print("tps template keys:", sorted(c.tps_settings["config_template"]))
print("splits/gen>0: frames_per_split =", c.dft_frames_per_split, "| cumulative =", c.cumulative_training,
      "| selection =", c.selection_mode, "| smd =", bool(c.smd_settings))
missing = [str(p) for p in list(c.initial_structures) + [Path(f) for f in c.dft_settings["input_files"]] + [c.initial_trajectory] if not Path(p).exists()]
sys.exit(f"MISSING: {missing}" if missing else print("all referenced files exist") or 0)
