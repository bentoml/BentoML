from __future__ import annotations

from pathlib import Path

import bentoml


@bentoml.service(workers=1)
class TaskService:
    @bentoml.task
    def shout(self, text: str) -> dict[str, str]:
        return {"echo": text.upper()}

    @bentoml.task
    def measure(self, blob: Path) -> int:
        return len(blob.read_text())
