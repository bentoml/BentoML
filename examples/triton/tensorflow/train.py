from __future__ import annotations

import tensorflow as tf
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

    bento_model_name = "tensorflow-yolov5"

    try:
        bentoml.models.get(bento_model_name)
        if args.override:
            raise bentoml.exceptions.NotFound(
                "'override=True', overriding previously saved weights/conversions."
            )
        print(f"{bento_model_name} already exists. Skipping...")
    except bentoml.exceptions.NotFound:
        _, metadata = load_traced_script()
        model = tf.saved_model.load(
            MODEL_FILE.__fspath__().replace(".pt", "_saved_model")
        )
        print(
            "Saved model:",
            bentoml.tensorflow.save_model(
                bento_model_name,
                model,
                signatures={"__call__": {"batchable": True, "batch_dim": (0, 0)}},
                tf_save_options=tf.saved_model.SaveOptions(
                    experimental_custom_gradients=False
                ),
                metadata={"model_info": metadata},
            ),
        )
    except Exception:
        print("Failed to save model:", bento_model_name)
        raise
