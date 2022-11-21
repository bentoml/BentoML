load("//rules/py:python.bzl", "framework_py_library")
load("@pypi//:requirements.bzl", "requirement")

exports_files(glob(["*.py"]))

framework_py_library(
    name = "catboost",
    deps = [requirement("scikit-learn")],
)

framework_py_library(
    name = "fastai",
    deps = [
        requirement("pandas"),
        requirement("scikit-learn"),
        requirement("numpy"),
    ],
)

framework_py_library(
    name = "lightgbm",
    deps = [
        requirement("pandas"),
        requirement("numpy"),
        requirement("scikit-learn"),
    ],
)

framework_py_library(
    name = "onnx",
    deps = [
        requirement("scikit-learn"),
        requirement("numpy"),
        requirement("torch"),
        requirement("skl2onnx"),
    ],
)

framework_py_library(name = "picklable_model")

framework_py_library(
    name = "pytorch",
    deps = [requirement("numpy")],
)

framework_py_library(
    name = "torchscript",
    deps = [
        requirement("numpy"),
        requirement("pandas"),
    ],
)

framework_py_library(
    name = "pytorch_lightning",
    deps = [
        "//:grpcio",
        requirement("numpy"),
    ],
)

framework_py_library(
    name = "sklearn",
    deps = [requirement("numpy")],
)

framework_py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    deps = [
        "//:grpcio",
        requirement("keras"),
        requirement("numpy"),
    ],
)

framework_py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        "//:grpcio",
        requirement("keras"),
        requirement("numpy"),
    ],
)

framework_py_library(
    name = "transformers",
    deps = [
        "//:grpcio",
        requirement("Pillow"),
        requirement("requests"),
    ],
)

framework_py_library(
    name = "xgboost",
    deps = [
        requirement("pandas"),
        requirement("numpy"),
        requirement("scikit-learn"),
    ],
)
