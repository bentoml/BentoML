load("//rules:python.bzl", "framework_py_library")
load("@pypi//:requirements.bzl", pypi = "requirement")
load("@frameworks//:requirements.bzl", framework = "requirement")

exports_files(glob(["*.py"]))

framework_py_library(
    name = "catboost",
    deps = [framework("scikit-learn")],
)

framework_py_library(
    name = "fastai",
    deps = [
        pypi("pandas"),
        framework("scikit-learn"),
        framework("numpy"),
    ],
)

framework_py_library(
    name = "lightgbm",
    deps = [
        pypi("pandas"),
        framework("numpy"),
        framework("scikit-learn"),
    ],
)

framework_py_library(
    name = "onnx",
    deps = [
        framework("scikit-learn"),
        framework("numpy"),
        framework("torch"),
        framework("skl2onnx"),
    ],
)

framework_py_library(name = "picklable_model")

framework_py_library(
    name = "pytorch",
    deps = [framework("numpy")],
)

framework_py_library(
    name = "torchscript",
    deps = [
        framework("numpy"),
        pypi("pandas"),
    ],
)

framework_py_library(
    name = "pytorch_lightning",
    deps = [framework("numpy")],
)

framework_py_library(
    name = "sklearn",
    deps = [framework("numpy")],
)

framework_py_library(
    name = "tensorflow",
    srcs = ["tensorflow.py"],
    deps = [
        framework("keras"),
        framework("numpy"),
    ],
)

framework_py_library(
    name = "keras",
    srcs = ["keras.py"],
    deps = [
        framework("keras"),
        framework("numpy"),
    ],
)

framework_py_library(
    name = "transformers",
    deps = [
        pypi("Pillow"),
        pypi("requests"),
    ],
)

framework_py_library(
    name = "xgboost",
    deps = [
        pypi("pandas"),
        framework("numpy"),
        framework("scikit-learn"),
    ],
)
