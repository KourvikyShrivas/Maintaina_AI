import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", ROOT / "knowledge"))
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "ai4i2020.csv"))
SEED = 42
