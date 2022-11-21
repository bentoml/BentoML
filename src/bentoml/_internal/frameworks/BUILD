load("@rules_python//python:defs.bzl", "py_library")
load("@frameworks//:requirements.bzl", framework = "requirement")

package(default_visibility = ["//src/bentoml:__pkg__"])

filegroup(
    name = "frameworks.template",
    srcs = ["FRAMEWORK_TEMPLATE_PY"],
    visibility = ["//src/bentoml/_internal/frameworks:__subpackages__"],
)

py_library(
    name = "catboost",
    srcs = ["catboost.py"],
    deps = [
        "//src/bentoml",
        "@pypi_attrs//:pkg",
        framework("numpy"),
        framework("pandas"),
        framework("catboost"),
    ],
)

py_library(
    name = "fastai",
    srcs = ["fastai.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        framework("torch"),
        framework("fastai"),
    ],
)

py_library(
    name = "lightgbm",
    srcs = ["lightgbm.py"],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            "//src/bentoml",
            framework("numpy"),
            framework("lightgbm"),
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//src/bentoml",
            "@frameworks_numpy//:pkg",
            "@com_github_microsoft_lightgbm//:python_package",
        ],
    }),
)

py_library(
    name = "mlflow",
    srcs = ["mlflow.py"],
    deps = [
        "//src/bentoml",
        framework("mlflow"),
    ],
)

py_library(
    name = "onnx",
    srcs = ["onnx.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/utils:onnx",
        framework("onnxruntime"),
        framework("protobuf"),
        framework("onnx"),
    ],
)

py_library(
    name = "picklable_model",
    srcs = ["picklable.py"],
    deps = [
        "//src/bentoml",
        "@pypi_cloudpickle//:pkg",
    ],
)

py_library(
    name = "pytorch",
    srcs = ["pytorch.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        "@pypi_cloudpickle//:pkg",
        framework("torch"),
    ],
)

py_library(
    name = "pytorch_lightning",
    srcs = ["pytorch_lightning.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        framework("pytorch-lightning"),
        framework("torch"),
    ],
)

py_library(
    name = "torchscript",
    srcs = ["torchscript.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        framework("torch"),
    ],
)

py_library(
    name = "sklearn",
    srcs = ["sklearn.py"],
    deps = [
        "//src/bentoml",
        framework("joblib"),
        framework("scikit-learn"),
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            framework("grpcio"),
            framework("keras"),
            framework("tensorflow"),
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            "//:grpcio",
            framework("keras"),
            framework("tensorflow-macos"),
        ],
    }),
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow_v2.py"],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            framework("grpcio"),
            framework("tensorflow"),
            "@pypi_attrs//:pkg",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            "//:grpcio",
            framework("tensorflow-macos"),
            "@pypi_attrs//:pkg",
        ],
    }),
)

py_library(
    name = "transformers",
    srcs = ["transformers.py"],
    deps = [
        "//src/bentoml",
        "@pypi_attrs//:pkg",
        framework("torch"),
        framework("flax"),
        framework("jax"),
        framework("transformers"),
    ],
)

py_library(
    name = "xgboost",
    srcs = ["xgboost.py"],
    deps = [
        "//src/bentoml",
        "@pypi_attrs//:pkg",
        framework("numpy"),
        framework("xgboost"),
    ],
)
