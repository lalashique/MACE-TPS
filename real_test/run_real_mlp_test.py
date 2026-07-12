"""Real-backend smoke test: actually trains a small MACE committee (via
mace.cli.run_train, CPU) on EMT-labeled Cu data using the ported
mlp/legacy_mace.py + mlp/mace_backend.py adapter, then runs real committee
uncertainty prediction. This is the genuine (non-mock) path through the
new package structure.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "packages/core")
sys.path.insert(0, "packages/mlp")

from core.interfaces import TrainedCommittee
from mlp.mace_backend import MACEBackend

E0s = {29: 0.0}  # Cu, approximate for this smoke test (EMT-labeled, not DFT)

backend = MACEBackend(e0s=E0s, config_file="real_test/mace_train_config.yaml")

print("=== training real MACE committee (seeds 123, 234) on CPU ===")
t0 = time.time()
committee = backend.train_committee(
    train_file=Path("real_test/train.xyz"),
    test_file=Path("real_test/test.xyz"),
    workdir=Path("real_test/runs"),
    seeds=[123, 234],
)
print(f"Training took {time.time() - t0:.1f}s")
print("Model paths:", committee.model_paths)
assert committee.size == 2, f"expected 2 committee members, got {committee.size}"
for p in committee.model_paths:
    assert Path(p).exists(), f"missing model file: {p}"

print("\n=== real committee uncertainty prediction ===")
results = backend.predict_uncertainty(committee, Path("real_test/candidates.xyz"))
for r in results:
    print(f"  frame {r.frame_index}: energy_var={r.energy_variance:.6f}  force_var={r.force_variance:.6f}")

print("\nREAL_MLP_TEST_PASSED")
