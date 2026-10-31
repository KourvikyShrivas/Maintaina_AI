"""Training script: python -m maintaina.train  (mirrors the Colab notebook, condensed)."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import MODEL_DIR, SEED
from .data import load_ai4i
from .features import CAT_COLS, ENG_COLS, FEATURES, NUM_COLS, TARGET, add_engineered_features

MODEL_FILE = "maintaina_failure_pipeline.joblib"
ANOMALY_FILE = "maintaina_anomaly_pipeline.joblib"
META_FILE = "maintaina_metadata.json"


def build_pipeline(model, engineered: bool) -> Pipeline:
    num = NUM_COLS + (ENG_COLS if engineered else [])
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])
    steps = [("features", FunctionTransformer(add_engineered_features, validate=False))] if engineered else []
    return Pipeline(steps + [("pre", pre), ("model", model)])


def _rf(n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2,
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)


def train(df: pd.DataFrame, model_dir: Path = MODEL_DIR, cost_fn: float = 10, cost_fp: float = 1,
          alert_rate: float = 0.03, n_estimators: int = 300) -> dict:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    X, y = df[FEATURES].copy(), df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    cv_scores = {eng: cross_val_score(build_pipeline(_rf(n_estimators), eng), X_tr, y_tr, cv=cv,
                                      scoring="average_precision", n_jobs=1).mean() for eng in (False, True)}
    engineered = bool(cv_scores[True] > cv_scores[False])
    model = build_pipeline(_rf(n_estimators), engineered)

    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    ths = np.linspace(0.01, 0.99, 99)
    yt = np.asarray(y_tr)
    costs = [cost_fn * int(((oof < t) & (yt == 1)).sum()) + cost_fp * int(((oof >= t) & (yt == 0)).sum()) for t in ths]
    threshold = float(ths[int(np.argmin(costs))])

    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {"pr_auc": average_precision_score(y_te, proba), "roc_auc": roc_auc_score(y_te, proba),
               "precision": precision_score(y_te, pred, zero_division=0), "recall": recall_score(y_te, pred),
               "f1": f1_score(y_te, pred), "test_failure_rate": float(y_te.mean()),
               "n_test": int(len(y_te)), "n_test_failures": int(y_te.sum())}

    iso = Pipeline([("scale", StandardScaler()),
                    ("iforest", IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1))]).fit(X_tr[NUM_COLS])
    cutoff = float(np.quantile(iso.decision_function(X_tr[NUM_COLS]), alert_rate))

    meta = {
        "project": "Maintaina AI",
        "dataset": "AI4I 2020 Predictive Maintenance Dataset (UCI id 601, synthetic, CC BY 4.0)",
        "input_schema": {c: ("categorical" if c in CAT_COLS else "numeric") for c in FEATURES},
        "categorical_values_seen": sorted(X_tr["type"].unique().tolist()),
        "numeric_ranges_in_training": {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in NUM_COLS},
        "numeric_mean": {c: float(X_tr[c].mean()) for c in NUM_COLS},
        "numeric_std": {c: float(X_tr[c].std()) for c in NUM_COLS},
        "uses_engineered_features": engineered,
        "cv_pr_auc_raw_vs_engineered": {"raw": float(cv_scores[False]), "engineered": float(cv_scores[True])},
        "failure_threshold": threshold,
        "cost_assumption": {"cost_false_negative": cost_fn, "cost_false_positive": cost_fp,
                            "note": "assumption, not measured from a real plant"},
        "anomaly": {"alert_rate": alert_rate, "score_cutoff": cutoff,
                    "note": "unusual-reading flag; NOT a failure probability"},
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "random_seed": SEED,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "limitations": ["Synthetic dataset; real-world performance unknown.",
                        "Tabular snapshots, no time series.",
                        "Decision-support only; not a replacement for inspection."],
    }
    joblib.dump(model, model_dir / MODEL_FILE)
    joblib.dump(iso, model_dir / ANOMALY_FILE)
    (model_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    a = ap.parse_args()
    meta = train(load_ai4i(a.data_path), Path(a.model_dir))
    print(json.dumps(meta["test_metrics"], indent=2))
    print("threshold:", meta["failure_threshold"], "| engineered:", meta["uses_engineered_features"])


if __name__ == "__main__":
    main()
