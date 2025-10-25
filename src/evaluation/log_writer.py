from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_jsonl_append(path: str | Path, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


def write_json_pretty(path: str | Path, record: Dict[str, Any]) -> None:
    """Write a pretty JSON file (one file per question or run). Overwrites existing."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
