#!/usr/bin/env python3
"""Check which parts of the MACE-TPS workbench are actually usable in the
current Python environment / PATH.

Nothing here is a hard requirement for the whole project - the workbench is
designed so each backend's dependencies are independent (see the main
README's package layout). This script just reports what it finds, with
install links for anything missing, so you know which backends
you can actually run without having to hit an ImportError partway through.

Run: python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import shutil
import sys


def ok(msg: str) -> None:
    print(f"  [ok]      {msg}")


def warn(msg: str) -> None:
    print(f"  [WARNING] {msg}")


def missing_module(name: str, import_name: str | None = None) -> bool:
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "?")
        ok(f"{name} ({version})")
        return False
    except ImportError:
        warn(f"{name} not importable")
        return True


def missing_binary(name: str, url: str, note: str = "") -> bool:
    path = shutil.which(name)
    if path:
        ok(f"{name} -> {path}")
        return False
    warn(f"{name} not found on PATH.")
    if note:
        print(f"            {note}")
    print(f"            Install: {url}")
    return True


def section(title: str) -> None:
    print(f"\n{title}")


def main() -> int:
    any_missing = False

    section("core (always required):")
    any_missing |= missing_module("core")

    section("mlp backend - mock (zero deps, always available):")
    ok("mlp.mock_backend")

    section("mlp backend - real (MACE training):")
    any_missing |= missing_module("ase")
    any_missing |= missing_module("torch")
    any_missing |= missing_module("mace-torch", "mace")

    section("dft backend - mock (zero deps, always available):")
    ok("dft.mock_backend")

    section("dft backend - PySCF (non-periodic molecules, pip-installable):")
    any_missing |= missing_module("pyscf")

    section("dft backend - CP2K (periodic slabs/solvated systems):")
    cp2k_missing = missing_binary(
        "cp2k.psmp",
        "https://www.cp2k.org/download",
        note=(
            "CP2K is a separate scientific-computing package (often built with MPI/OpenMP for your "
            "specific platform) - it is not installed by this project's environment.yml or pip "
            "extras. Install it natively (conda-forge also has a 'cp2k' package, or build from "
            "source), then make sure the binary you configure in dft_settings['command'] is on PATH "
            "or given as an absolute path."
        ),
    )
    if cp2k_missing:
        # also check the plain "cp2k" name some builds use instead of "cp2k.psmp"
        missing_binary("cp2k", "https://www.cp2k.org/download")
    any_missing |= cp2k_missing

    section("tps backend - mock (zero deps, always available):")
    ok("tps.mock_backend")

    section("tps backend - real (OpenPathSampling stack, conda-forge only):")
    tps_stack_missing = [
        missing_module("openpathsampling"),
        missing_module("openmm"),
        missing_module("openmmml"),
        missing_module("mdtraj"),
    ]
    any_missing |= any(tps_stack_missing)
    if any(tps_stack_missing):
        print(
            "            This stack isn't reliably pip-installable together - see environment.yml "
            "and the main README for the conda/mamba setup."
        )

    section("notebook execution (for analysis notebooks):")
    any_missing |= missing_module("nbformat")
    any_missing |= missing_module("nbclient")
    any_missing |= missing_module("matplotlib")

    print()
    if any_missing:
        print("Some optional components are missing - see warnings above. The mock backends and")
        print("core workflow always work regardless; missing items only limit which real backends")
        print("you can run.")
    else:
        print("Everything checked is available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
