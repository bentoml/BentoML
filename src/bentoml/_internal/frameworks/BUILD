load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", "requirement")
load("@bazel_skylib//lib:selects.bzl", "selects")

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
        requirement("attrs"),
        requirement("numpy"),
        requirement("pandas"),
        requirement("catboost"),
    ],
)

py_library(
    name = "fastai",
    srcs = ["fastai.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        requirement("torch"),
        requirement("fastai"),
    ],
)

py_library(
    name = "lightgbm",
    srcs = ["lightgbm.py"],
    deps = selects.with_or({
        ("@bazel_tools//src/conditions:linux", "@bazel_tools//src/conditions:windows"): [
            "//src/bentoml",
            requirement("numpy"),
            requirement("lightgbm"),
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//src/bentoml",
            "@com_github_microsoft_lightgbm//:python_package",
            requirement("numpy"),
        ],
    }),
)

py_library(
    name = "mlflow",
    srcs = ["mlflow.py"],
    deps = [
        "//src/bentoml",
        requirement("mlflow"),
    ],
)

py_library(
    name = "onnx",
    srcs = ["onnx.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/utils:onnx",
        requirement("onnxruntime"),
        requirement("protobuf"),
        requirement("onnx"),
    ],
)

py_library(
    name = "picklable_model",
    srcs = ["picklable.py"],
    deps = [
        "//src/bentoml",
        requirement("cloudpickle"),
    ],
)

py_library(
    name = "pytorch",
    srcs = ["pytorch.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        requirement("cloudpickle"),
        requirement("torch"),
    ],
)

py_library(
    name = "pytorch_lightning",
    srcs = ["pytorch_lightning.py"],
    deps = [
        "//:grpcio",
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        requirement("torch"),
        requirement("lightning-lite"),
    ],
)

py_library(
    name = "torchscript",
    srcs = ["torchscript.py"],
    deps = [
        "//src/bentoml",
        "//src/bentoml/_internal/frameworks/common:pytorch",
        requirement("torch"),
    ],
)

py_library(
    name = "sklearn",
    srcs = ["sklearn.py"],
    deps = [
        "//src/bentoml",
        requirement("joblib"),
        requirement("scikit-learn"),
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = selects.with_or({
        "@bazel_tools//src/conditions:darwin_arm64": [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            "//:grpcio",
            requirement("keras"),
            requirement("tensorflow-macos"),
        ],
        ("@bazel_tools//src/conditions:linux", "@bazel_tools//src/conditions:windows", "@bazel_tools//src/conditions:darwin_x86_64"): [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            "//:grpcio",
            requirement("keras"),
            requirement("tensorflow"),
        ],
    }),
)

py_library(
    name = "tensorflow",
    srcs = ["tensorflow_v2.py"],
    deps = selects.with_or({
        "@bazel_tools//src/conditions:darwin_arm64": [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            "//:grpcio",
            requirement("tensorflow-macos"),
            requirement("attrs"),
        ],
        ("@bazel_tools//src/conditions:linux", "@bazel_tools//src/conditions:windows", "@bazel_tools//src/conditions:darwin_x86_64"): [
            "//src/bentoml",
            "//src/bentoml/_internal/frameworks/utils:tensorflow",
            "//:grpcio",
            requirement("tensorflow"),
            requirement("attrs"),
        ],
    }),
)

py_library(
    name = "transformers",
    srcs = ["transformers.py"],
    deps = selects.with_or({
        "@bazel_tools//src/conditions:darwin_arm64": [
            "//src/bentoml",
            requirement("attrs"),
            requirement("tensorflow-macos"),
            requirement("torch"),
            requirement("flax"),
            requirement("jax"),
            requirement("transformers"),
        ],
        ("@bazel_tools//src/conditions:linux", "@bazel_tools//src/conditions:darwin_x86_64"): [
            "//src/bentoml",
            requirement("attrs"),
            requirement("tensorflow"),
            requirement("torch"),
            requirement("flax"),
            requirement("jax"),
            requirement("transformers"),
        ],
        "@bazel_tools//src/conditions:windows": [
            "//src/bentoml",
            requirement("attrs"),
            requirement("tensorflow"),
            requirement("torch"),
            requirement("transformers"),
        ],
    }),
)

py_library(
    name = "xgboost",
    srcs = ["xgboost.py"],
    deps = [
        "//src/bentoml",
        requirement("attrs"),
        requirement("numpy"),
        requirement("xgboost"),
    ],
)
