import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from .config import DATA_PATH
from .features import clean_name

DATA_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def load_ai4i(path: Path | None = None) -> pd.DataFrame:
    """Load AI4I 2020 from a local CSV, downloading it from UCI first if needed."""
    path = Path(path or DATA_PATH)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            members = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not members:
                raise RuntimeError(f"No CSV in downloaded zip: {z.namelist()}")
            path.write_bytes(z.read(members[0]))
    df = pd.read_csv(path)
    df.columns = [clean_name(c) for c in df.columns]
    return df
