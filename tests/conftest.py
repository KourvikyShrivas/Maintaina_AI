import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from maintaina.train import train


def make_fake_ai4i(n=2000, seed=0) -> pd.DataFrame:
    """Synthetic stand-in with the AI4I schema, ONLY for tests (never used for reported results)."""
    r = np.random.default_rng(seed)
    t = r.choice(list("LMH"), n, p=[.6, .3, .1])
    air = r.normal(300, 2, n); proc = air + 10 + r.normal(0, 1, n)
    rpm = r.normal(1540, 170, n).astype(int).clip(1000); tq = r.normal(40, 10, n).clip(3)
    wear = r.integers(0, 250, n)
    fail = ((wear * tq > 11000) | (r.random(n) < 0.01)).astype(int)
    return pd.DataFrame({"udi": range(n), "product_id": [f"{a}{i}" for i, a in enumerate(t)], "type": t,
                         "air_temperature": air, "process_temperature": proc, "rotational_speed": rpm,
                         "torque": tq, "tool_wear": wear, "machine_failure": fail,
                         "twf": 0, "hdf": 0, "pwf": 0, "osf": 0, "rnf": 0})


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    train(make_fake_ai4i(), d, n_estimators=30)
    return d


@pytest.fixture()
def knowledge_dir(tmp_path):
    (tmp_path / "test_manual.txt").write_text(
        "TEST FIXTURE, not a real manual. Inspect the cutting tool for wear before replacement. "
        "Tool wear above the limit requires replacing the tool insert. " * 5 +
        "Unrelated cafeteria menu: soup and sandwich on Tuesday. " * 5)
    return tmp_path


@pytest.fixture()
def client(model_dir, knowledge_dir):
    with TestClient(create_app(model_dir, knowledge_dir)) as c:
        yield c


@pytest.fixture()
def client_no_model(tmp_path, knowledge_dir):
    with TestClient(create_app(tmp_path / "empty", knowledge_dir)) as c:
        yield c
