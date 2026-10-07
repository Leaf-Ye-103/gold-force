import pytest
from gold_force.demo import build_demo


@pytest.fixture
def weekly(tmp_path):
    cfg, ledger = build_demo(tmp_path / "weekly")
    cfg["bootstrap"]["iterations"] = 0
    cfg["model"] = {"window": 26, "min_train": 16, "alphas": [1.0], "cv_splits": 3}
    cfg["end"] = "2019-01-01"
    return cfg, ledger.loc[ledger.observation_date.le(cfg["end"])].copy()
