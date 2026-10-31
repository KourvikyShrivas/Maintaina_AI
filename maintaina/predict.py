import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR
from .features import FEATURES, NUM_COLS
from .train import ANOMALY_FILE, META_FILE, MODEL_FILE

log = logging.getLogger("maintaina")

DISCLAIMER = ("Decision-support only. Scores come from a model trained on a synthetic dataset and are not "
              "guaranteed to reflect a real machine. An anomaly flag means 'unusual', not 'failing'. "
              "Always follow qualified inspection procedures.")


class ModelNotReady(RuntimeError):
    pass


class PredictionService:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model = self.anomaly = self.meta = None
        self.load()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        try:
            self.model = joblib.load(self.model_dir / MODEL_FILE)
            self.anomaly = joblib.load(self.model_dir / ANOMALY_FILE)
            self.meta = json.loads((self.model_dir / META_FILE).read_text())
        except FileNotFoundError:
            self.model = self.anomaly = self.meta = None
            log.warning("Model artifacts not found in %s; run `python -m maintaina.train`", self.model_dir)

    def predict(self, reading: dict) -> dict:
        if not self.ready:
            raise ModelNotReady("Model artifacts are not available. Train the model first.")
        m = self.meta
        row = pd.DataFrame([reading])[FEATURES]
        proba = float(self.model.predict_proba(row)[0, 1])
        thr = m["failure_threshold"]
        score = float(self.anomaly.decision_function(row[NUM_COLS])[0])
        anomaly = score < m["anomaly"]["score_cutoff"]

        warnings, unusual = [], []
        for c in NUM_COLS:
            lo, hi = m["numeric_ranges_in_training"][c]
            v = float(row[c].iloc[0])
            if v < lo or v > hi:
                warnings.append(f"{c}={v} is outside the training range [{lo:.1f}, {hi:.1f}]; "
                                "the model is extrapolating and the score is less reliable.")
            z = (v - m["numeric_mean"][c]) / (m["numeric_std"][c] or 1.0)
            if abs(z) >= 2:
                unusual.append({"feature": c, "value": v, "z_score": round(z, 2)})
        if float(row["process_temperature"].iloc[0]) <= float(row["air_temperature"].iloc[0]):
            warnings.append("process_temperature is not above air_temperature; please double-check the sensors.")
        unusual.sort(key=lambda d: -abs(d["z_score"]))

        log.info("prediction alert=%s anomaly=%s proba=%.3f", proba >= thr, anomaly, proba)  # no raw inputs logged
        return {"failure_probability": round(proba, 4), "alert": proba >= thr, "threshold": round(thr, 4),
                "anomaly": bool(anomaly), "anomaly_score": round(score, 4),
                "unusual_inputs": unusual, "warnings": warnings, "disclaimer": DISCLAIMER}
