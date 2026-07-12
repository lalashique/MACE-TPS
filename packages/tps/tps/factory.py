from __future__ import annotations

from core.interfaces import TPSSampler


def create_tps_sampler(backend: str, settings: dict) -> TPSSampler:
    if backend == "mock":
        from tps.mock_backend import MockTPSSampler

        return MockTPSSampler(
            jitter=settings.get("jitter", 0.05), seed=settings.get("seed", 0)
        )
    if backend == "ops":
        from tps.ops_backend import OPSTPSSampler

        return OPSTPSSampler(config_template=settings.get("config_template", {}))
    raise ValueError(f"Unknown TPS backend: {backend}")
