from pathlib import Path
import shutil
from .config import guarded, digest, write_json
from .scenario_table import ScenarioTable

def import_frozen(source, provenance):
    if not provenance.strip():
        raise ValueError("External data requires provenance")
    source = Path(source).resolve()
    table = ScenarioTable.load(source)
    destination = guarded(f"data/external_frozen/{digest(source)}.npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copyfile(source, destination)
    write_json(destination.with_suffix(".json"), {"source": str(source), "provenance": provenance,
               "sha256": digest(destination), "table_sha256": table.sha256, "schema_version": 1,
               "scope": "external_interface_extension_only"})
    return destination
