from __future__ import annotations

import os
import typing as t
import argparse

import attr
import torch

import bentoml
from bentoml import Bento
from bentoml._internal.utils import resolve_user_filepath
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.configuration.containers import BentoMLContainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--gpu", action="store_true", default=False)

    args = parser.parse_args()

    bento_tag = "triton-integration-onnx"
    if args.tag:
        bento_tag = f"triton-integration-onnx:{args.tag}"

    try:
        bentos = bentoml.get(bento_tag)
        print(f"{bentos} already exists. Skipping...")
    except bentoml.exceptions.NotFound:
        bentofile = resolve_user_filepath("bentofile.yaml", None)

        override_attrs: dict[str, t.Any] = {
            "python": {
                "requirements_txt": os.path.join("requirements", "requirements-gpu.txt")
                if args.gpu and torch.cuda.is_available()
                else os.path.join("requirements", "requirements.txt")
            }
        }
        with open(bentofile, "r", encoding="utf-8") as f:
            build_config = attr.evolve(BentoBuildConfig.from_yaml(f), **override_attrs)

        print(
            "Saved bento:",
            Bento.create(build_config, version=args.tag).save(
                BentoMLContainer.bento_store.get()
            ),
        )
