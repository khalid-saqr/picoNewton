"""Validation for committed CellML source assets."""

from __future__ import annotations

from pathlib import Path
import json
import xml.etree.ElementTree as ET

from .provenance import sha256_file


def cellml_root(package_root: Path) -> Path:
    return package_root / "external" / "cellml" / "ogiermann_2025"


def validate_cellml_sources(package_root: Path) -> dict[str, object]:
    root = cellml_root(package_root)
    source = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    records = []
    failures = []
    for path in sorted(root.rglob("*.cellml")):
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            failures.append(f"xml:{path.name}:{exc}")
            continue
        records.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    required = {"units.cellml", "models/channels/ogiermann_2025_piezo1.cellml", "models/channels/ogiermann_2025_piezo1_params.cellml"}
    found = {record["path"] for record in records}
    for missing in sorted(required - found):
        failures.append(f"missing:{missing}")
    return {"ok": not failures, "failures": failures, "license": source["license"], "workspace_revision": source["workspace_revision"], "files": records}
