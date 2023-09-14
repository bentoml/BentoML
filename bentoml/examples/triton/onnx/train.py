from __future__ import annotations

import onnx
import onnx.checker as onnx_checker
from helpers import MODEL_FILE
from helpers import load_traced_script

import bentoml

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

    bento_model_name = "onnx-yolov5"
    _, metadata = load_traced_script()
    d = {"stride": metadata["stride"], "names": metadata["names"]}
    try:
        bentoml.models.get(bento_model_name)
        if args.override:
            raise bentoml.exceptions.NotFound(
                "'override=True', overriding previously saved weights/conversions."
            )
        print(f"{bento_model_name} already exists. Skipping...")
    except bentoml.exceptions.NotFound:
        ModelProto = onnx.load(MODEL_FILE.with_suffix(".onnx").__fspath__())
        onnx_checker.check_model(ModelProto)
        for k, v in d.items():
            meta = ModelProto.metadata_props.add()
            meta.key, meta.value = k, str(v)
        print(
            "Saved model:",
            bentoml.onnx.save_model(
                bento_model_name,
                ModelProto,
                signatures={"run": {"batchable": True, "batch_dim": (0, 0)}},
                metadata={"model_info": d},
            ),
        )
    except Exception:
        print("Failed to save model:", bento_model_name)
        raise
