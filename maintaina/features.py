"""Column definitions and feature engineering. Lives in a module so saved pipelines can be unpickled anywhere."""
import re

import numpy as np
import pandas as pd

TARGET = "machine_failure"
ID_COLS = ["udi", "product_id"]
MODE_COLS = ["twf", "hdf", "pwf", "osf", "rnf"]  # failure modes: leakage if used as inputs
NUM_COLS = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
CAT_COLS = ["type"]
FEATURES = NUM_COLS + CAT_COLS
ENG_COLS = ["temp_diff", "power_w", "wear_x_torque"]


def clean_name(c: str) -> str:
    c = re.sub(r"\[.*?\]", "", c)
    c = re.sub(r"[^0-9a-zA-Z]+", "_", c.strip())
    return c.strip("_").lower()


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["temp_diff"] = X["process_temperature"] - X["air_temperature"]
    X["power_w"] = X["torque"] * X["rotational_speed"] * 2 * np.pi / 60.0
    X["wear_x_torque"] = X["tool_wear"] * X["torque"]
    return X
