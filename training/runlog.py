"""JSONL experiment logs. Weight change is recorded; it is not a learning claim."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            **record,
        }
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
