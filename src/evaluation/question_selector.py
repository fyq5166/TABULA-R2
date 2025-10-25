"""
Question selection utilities for Q&A evaluation.

Supports selection modes:
- all: all questions found under the provided root
- single_table: questions under data/questions/single_table
- multi_table: questions under data/questions/multi_table/**/question_*.json
- distractor: questions under data/questions/distractor_bank/**/question_*.json
- custom: parse explicit ID specs like ["1", "2-3", "7"]

Outputs randomized question IDs for testing.
"""

from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Any

from ..utils.io_utils import safe_load_json  # using utils as requested

# Note: intentionally avoiding package-relative imports here so the module
# can be executed directly via `python src/evaluation/question_selector.py`.


@dataclass
class SelectionConfig:
    mode: str  # all | single_table | multi_table | distractor | custom
    root: Path
    custom_specs: List[str]
    seed: int


def _find_json_files(directory: Path) -> Iterable[Path]:
    if not directory.exists():
        return []
    return directory.rglob("question_*.json")


def _extract_id_from_filename(path: Path) -> int | None:
    m = re.search(r"question_(\d+)\.json", path.name)
    return int(m.group(1)) if m else None


def _expand_id_specs(specs: List[str]) -> List[int]:
    ids: List[int] = []
    for spec in specs:
        for token in spec.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                a, b = token.split("-", 1)
                try:
                    start = int(a)
                    end = int(b)
                except ValueError:
                    continue
                if start <= end:
                    ids.extend(range(start, end + 1))
                else:
                    ids.extend(range(end, start + 1))
            else:
                try:
                    ids.append(int(token))
                except ValueError:
                    continue
    # de-duplicate while preserving order
    seen = set()
    ordered: List[int] = []
    for qid in ids:
        if qid not in seen:
            seen.add(qid)
            ordered.append(qid)
    return ordered


def collect_question_ids(cfg: SelectionConfig) -> List[int]:
    mode = cfg.mode.lower()
    if mode == "single_table":
        sources = list(_find_json_files(cfg.root / "single_table"))
    elif mode == "multi_table":
        sources = list(_find_json_files(cfg.root / "multi_table"))
    elif mode == "distractor":
        sources = list(_find_json_files(cfg.root / "distractor_bank"))
    elif mode == "all":
        sources = []
        for sub in ("single_table", "multi_table", "distractor_bank"):
            sources.extend(list(_find_json_files(cfg.root / sub)))
    elif mode == "custom":
        # Build mapping of existing question IDs to paths
        id_to_paths: Dict[int, List[Path]] = {}
        for sub in ("single_table", "multi_table", "distractor_bank"):
            for p in _find_json_files(cfg.root / sub):
                qid = _extract_id_from_filename(p)
                if qid is not None:
                    id_to_paths.setdefault(qid, []).append(p)

        # Expand specs and filter to existing IDs
        expanded = _expand_id_specs(cfg.custom_specs)
        filtered = [qid for qid in expanded if qid in id_to_paths]

        # Randomize for testing
        rnd = random.Random(cfg.seed)
        rnd.shuffle(filtered)
        return filtered
    else:
        raise ValueError(f"Unknown selection mode: {cfg.mode}")

    ids = [qid for p in sources if (qid := _extract_id_from_filename(p)) is not None]
    # randomize order for testing
    rnd = random.Random(cfg.seed)
    rnd.shuffle(ids)
    return ids


def collect_question_records(cfg: SelectionConfig) -> List[Dict[str, Any]]:
    """Return question records with question_id, table_refs, question, answer."""
    mode = cfg.mode.lower()
    sources: List[Path]
    if mode in {"single_table", "multi_table", "distractor", "all"}:
        if mode == "single_table":
            sources = list(_find_json_files(cfg.root / "single_table"))
        elif mode == "multi_table":
            sources = list(_find_json_files(cfg.root / "multi_table"))
        elif mode == "distractor":
            sources = list(_find_json_files(cfg.root / "distractor_bank"))
        else:
            sources = []
            for sub in ("single_table", "multi_table", "distractor_bank"):
                sources.extend(list(_find_json_files(cfg.root / sub)))
        items: List[tuple[int, Path]] = []
        for p in sources:
            qid = _extract_id_from_filename(p)
            if qid is not None:
                items.append((qid, p))
    elif mode == "custom":
        # map id->paths
        id_to_paths: Dict[int, List[Path]] = {}
        for sub in ("single_table", "multi_table", "distractor_bank"):
            for p in _find_json_files(cfg.root / sub):
                qid = _extract_id_from_filename(p)
                if qid is not None:
                    id_to_paths.setdefault(qid, []).append(p)
        wanted = _expand_id_specs(cfg.custom_specs)
        items = []
        for qid in wanted:
            paths = id_to_paths.get(qid)
            if paths:
                # Prefer the first path if duplicates across categories
                items.append((qid, paths[0]))
    else:
        raise ValueError(f"Unknown selection mode: {cfg.mode}")

    # Randomize by seed
    rnd = random.Random(cfg.seed)
    rnd.shuffle(items)

    records: List[Dict[str, Any]] = []
    for qid, path in items:
        data = safe_load_json(path, default={})
        record = {
            "question_id": qid,
            "table_refs": data.get("table_refs", []),
            "question": data.get("question", None),
            "answer": data.get("answer", None),
        }
        records.append(record)
    return records


if __name__ == "__main__":
    # Tester removed after verification
    pass
