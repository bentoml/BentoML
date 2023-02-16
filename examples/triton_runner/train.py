from __future__ import annotations

import os
import sys
import json
import typing as t
import subprocess

import fs
import fs.copy
import fs.mirror
from helpers import tf
from helpers import onnx
from helpers import torch
from helpers import MODEL_FILE
from helpers import onnx_checker
from helpers import TORCH_DEVICE
from helpers import ModelRepository
from helpers import safe_create_bentomodel

import bentoml

if t.TYPE_CHECKING:
    from torch.jit._script import ScriptModule


def load_traced_script() -> tuple[ScriptModule, dict[str, t.Any]]:
    extra_files = {"config.txt": ""}
    model = torch.jit.load(
        MODEL_FILE.with_suffix(".torchscript").__fspath__(),
        map_location=TORCH_DEVICE,
        _extra_files=extra_files,
    )
    d = {"shape": [], "stride": 32, "names": {}}
    if extra_files["config.txt"]:
        d = json.loads(
            extra_files["config.txt"],
            object_hook=lambda d: {
                int(k) if k.isdigit() else k: v for k, v in d.items()
            },
        )
    return model, d


@safe_create_bentomodel
def onnx_yolov5(**kwargs: t.Any) -> bentoml.Model:
    bento_model_name: str = kwargs.pop("bento_model_name")
    _, metadata = load_traced_script()
    d = {"stride": metadata["stride"], "names": metadata["names"]}
    ModelProto = onnx.load(MODEL_FILE.with_suffix(".onnx").__fspath__())
    onnx_checker.check_model(ModelProto)
    for k, v in d.items():
        meta = ModelProto.metadata_props.add()
        meta.key, meta.value = k, str(v)
    return bentoml.onnx.save_model(
        bento_model_name,
        ModelProto,
        signatures={"run": {"batchable": True, "batch_dim": (0, 0)}},
        metadata={"model_info": d},
    )


def onnx_mnist_triton():
    m = bentoml.onnx.get("onnx-mnist")
    triton_mnist = ModelRepository.getmodelfs("onnx_mnist")
    fs.copy.copy_file(m._fs, "saved_model.onnx", triton_mnist, "model.onnx")
    print("Created weights for MNIST model:", triton_mnist.getsyspath("/"))


@safe_create_bentomodel
def torchscript_yolov5(**kwargs: t.Any) -> bentoml.Model:
    bento_model_name: str = kwargs.pop("bento_model_name")
    model, extra_files = load_traced_script()
    return bentoml.torchscript.save_model(
        bento_model_name,
        model,
        signatures={"__call__": {"batchable": True, "batch_dim": (0, 0)}},
        metadata={"model_info": extra_files},
        _extra_files={"config.txt": json.dumps(extra_files)},
    )


def torchscript_mnist_triton():
    m = bentoml.torchscript.get("torchscript-mnist")
    triton_mnist = ModelRepository.getmodelfs("torchscript_mnist")
    fs.copy.copy_file(m._fs, "saved_model.pt", triton_mnist, "model.pt")
    print("Created weights for MNIST model:", triton_mnist.getsyspath("/"))


@safe_create_bentomodel
def tensorflow_yolov5(**kwargs: t.Any) -> bentoml.Model:
    bento_model_name: str = kwargs.pop("bento_model_name")

    _, metadata = load_traced_script()
    model = tf.saved_model.load(MODEL_FILE.__fspath__().replace(".pt", "_saved_model"))
    return bentoml.tensorflow.save_model(
        bento_model_name,
        model,
        signatures={"__call__": {"batchable": True, "batch_dim": (0, 0)}},
        tf_save_options=tf.saved_model.SaveOptions(experimental_custom_gradients=False),
        metadata={"model_info": metadata},
    )


def tensorflow_mnist_triton():
    m = bentoml.tensorflow.get("tensorflow-mnist")
    triton_mnist = ModelRepository.getmodelfs("tensorflow_mnist")
    triton_mnist.makedirs("model.savedmodel", recreate=True)
    fs.mirror.mirror(m._fs, fs.open_fs(triton_mnist.getsyspath("/model.savedmodel")))
    print("Created weights for MNIST model:", triton_mnist.getsyspath("/"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="Override existing models",
    )
    parser.add_argument(
        "--opset-version", type=int, default=17, help="ONNX opset version"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Max batch size")

    args = parser.parse_args()

    for framework in ["onnx", "torchscript", "tensorflow"]:
        globals()[f"{framework}_yolov5"](**vars(args))
        arguments = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "neuralnet", f"_{framework}.py"),
            "--batch-size",
            str(args.batch_size),
        ]
        if framework == "onnx":
            arguments.extend(["--opset-version", str(args.opset_version)])
        subprocess.check_call(arguments)
        # export MNIST models to triton Runner
        globals()[f"{framework}_mnist_triton"]()
