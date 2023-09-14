from __future__ import annotations

import json
import typing as t

from helpers import load_traced_script

import bentoml

if t.TYPE_CHECKING:
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Override existing models",
    )
    args = parser.parse_args()

    bento_model_name = "torchscript-yolov5"
    model, metadata = load_traced_script()

    try:
        bentoml.models.get(bento_model_name)
        if args.override:
            raise bentoml.exceptions.NotFound(
                "'override=True', overriding previously saved weights/conversions."
            )
        print(f"{bento_model_name} already exists. Skipping...")
    except bentoml.exceptions.NotFound:
        print(
            "Saved model:",
            bentoml.torchscript.save_model(
                bento_model_name,
                model,
                signatures={"__call__": {"batchable": True, "batch_dim": (0, 0)}},
                metadata={"model_info": metadata},
                _extra_files={"config.txt": json.dumps(metadata)},
            ),
        )
    except Exception:
        print("Failed to save model:", bento_model_name)
        raise
