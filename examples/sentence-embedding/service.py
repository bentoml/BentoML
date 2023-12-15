from __future__ import annotations

import os

import numpy as np
import torch

import bentoml
from bentoml.ui import Field


@bentoml.service(resources={"memory": "500MiB"}, traffic={"timeout": 1})
class SentenceEmbedding:
    model_ref = bentoml.models.get("all-MiniLM-L6-v2")

    def __init__(self) -> None:
        from transformers import AutoModel
        from transformers import AutoTokenizer

        print("Init", self.model_ref.path)
        # Load model and tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_ref.path_of("/")).to(
            self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ref.path_of("/"))
        print("Model loaded", "device:", self.device)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @bentoml.api(batchable=True)
    def encode(
        self,
        sentences: list[str] = Field(["hello world"], description="input sentences"),
    ) -> np.ndarray:
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Optional: Normalize embeddings if needed
        if str(os.getenv("NORMALIZE", False)).upper() in [
            "TRUE",
            "1",
            "YES",
            "Y",
            "ON",
        ]:
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )

        return sentence_embeddings.cpu().numpy()
