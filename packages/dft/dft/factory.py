from __future__ import annotations

from core.interfaces import DFTEngine


def create_dft_engine(backend: str, settings: dict) -> DFTEngine:
    if backend == "mock":
        from dft.mock_backend import MockDFTEngine

        return MockDFTEngine(
            energy_key=settings.get("energy_key", "energy_cp2k"),
            forces_key=settings.get("forces_key", "forces_cp2k"),
        )
    if backend == "cp2k":
        from dft.cp2k_backend import CP2KDFTEngine

        return CP2KDFTEngine(
            settings=settings.get("settings", {}),
            input_files=settings.get("input_files", []),
        )
    if backend == "pyscf":
        from dft.pyscf_backend import PySCFDFTEngine

        return PySCFDFTEngine(
            basis=settings.get("basis", "6-31g"),
            xc=settings.get("xc", "b3lyp"),
            charge=settings.get("charge", 0),
            spin=settings.get("spin", 0),
            energy_key=settings.get("energy_key", "energy_cp2k"),
            forces_key=settings.get("forces_key", "forces_cp2k"),
        )
    raise ValueError(f"Unknown DFT backend: {backend}")
