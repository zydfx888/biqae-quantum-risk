"""Explicit, project-local public data download with provenance."""
from urllib.request import urlopen
from .config import guarded, digest, write_json

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/5_Industry_Portfolios_daily_CSV.zip"

def download_french(destination="data/raw/5_Industry_Portfolios_daily_CSV.zip"):
    path = guarded(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace frozen input: {path}")
    with urlopen(FRENCH_URL, timeout=60) as response:
        content = response.read()
    path.write_bytes(content)
    write_json(path.with_suffix(".manifest.json"), {"source": FRENCH_URL, "sha256": digest(path)})
    return path
