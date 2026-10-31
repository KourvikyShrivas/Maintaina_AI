import json

import joblib
import numpy as np
import pandas as pd

from maintaina.features import FEATURES
from maintaina.predict import ModelNotReady, PredictionService
from maintaina.train import META_FILE, MODEL_FILE
import pytest

ROW = {"type": "M", "air_temperature": 300, "process_temperature": 310,
       "rotational_speed": 1500, "torque": 40, "tool_wear": 100}


def test_service_predicts(model_dir):
    out = PredictionService(model_dir).predict(ROW)
    assert 0 <= out["failure_probability"] <= 1


def test_reload_is_identical(model_dir):
    a = joblib.load(model_dir / MODEL_FILE)
    b = joblib.load(model_dir / MODEL_FILE)
    X = pd.DataFrame([ROW])[FEATURES]
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_metadata_has_schema_and_threshold(model_dir):
    m = json.loads((model_dir / META_FILE).read_text())
    assert 0 < m["failure_threshold"] < 1 and set(m["input_schema"]) == set(FEATURES)
    assert "limitations" in m


def test_no_leakage_columns_in_inputs():
    assert not {"twf", "hdf", "pwf", "osf", "rnf", "machine_failure", "udi", "product_id"} & set(FEATURES)


def test_not_ready_raises(tmp_path):
    with pytest.raises(ModelNotReady):
        PredictionService(tmp_path).predict(ROW)
