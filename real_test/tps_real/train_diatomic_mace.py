import sys
from pathlib import Path

sys.path.insert(0, "packages/core")
sys.path.insert(0, "packages/mlp")

from mlp.mace_backend import MACEBackend

backend = MACEBackend(e0s={1: 0.0}, config_file="real_test/tps_real/mace_train_config.yaml")
committee = backend.train_committee(
    train_file=Path("real_test/tps_real/train.xyz"),
    test_file=Path("real_test/tps_real/test.xyz"),
    workdir=Path("real_test/tps_real/runs"),
    seeds=[123, 234, 345],
)
print("Trained model paths:")
for p in committee.model_paths:
    print(" ", p)

import json

tps_json = Path("real_test/tps_real/tps.json")
config = json.loads(tps_json.read_text())
config["mace_model_path"] = [str(p) for p in committee.model_paths]
tps_json.write_text(json.dumps(config, indent=4))
print(f"\nUpdated {tps_json} with real trained model paths.")
print("TRAIN_DIATOMIC_DONE")
