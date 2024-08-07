from __future__ import annotations

from typing import List

import bentoml


@bentoml.service(resources={"cpu": "4"})
class Summarization:
    def __init__(self) -> None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = pipeline("summarization", device=device)

    @bentoml.api(batchable=True)
    def summarize(self, texts: List[str]) -> List[str]:
        results = self.pipeline(texts)
        return [item["summary_text"] for item in results]
