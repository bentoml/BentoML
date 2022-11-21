load("//rules:python.bzl", "py_test")
load("@rules_python//python:defs.bzl", "py_library")
load("@pypi//:requirements.bzl", "requirement")

filegroup(
    name = "config",
    srcs = [
        "bentofile.yaml",
        "requirements.txt",
    ],
)

filegroup(
    name = "conftest",
    srcs = ["tests/conftest.py"],
)

filegroup(
    name = "srcs_files",
    srcs = glob(["*.py"]),
)

py_library(
    name = "python_model",
    srcs = ["python_model.py"],
    deps = [
        "//src/bentoml:picklable_model",
        requirement("pandas"),
        requirement("numpy"),
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":python_model",
        "//src/bentoml",
        "//tests:tests_lib",
    ],
)

py_library(
    name = "context_server_interceptor",
    srcs = ["context_server_interceptor.py"],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            requirement("grpcio"),
            "//src/bentoml",
            "//tests:tests_lib",
        ],
        "@bazel_tools//src/conditions:darwin": [
            "//:grpcio",
            "//src/bentoml",
            "//tests:tests_lib",
        ],
    }),
)

py_test(
    name = "e2e",
    srcs = ["tests/test_{}.py".format(lib) for lib in [
        "custom_components",
        "custom_metrics_registration",
        "descriptors",
    ]] + glob(["*.py"]),
    data = [
        ":config",
        ":conftest",
        ":srcs_files",
    ],
    deps = select({
        "@bazel_tools//src/conditions:linux": [
            ":context_server_interceptor",
            ":train",
            requirement("grpcio"),
            requirement("protobuf"),
            requirement("grpcio-health-checking"),
            requirement("numpy"),
            requirement("pydantic"),
            requirement("pandas"),
            requirement("Pillow"),
        ],
        "@bazel_tools//src/conditions:darwin": [
            ":context_server_interceptor",
            ":train",
            "//:grpcio",
            "//:grpcio-health-checking",
            "//:protobuf",
            requirement("numpy"),
            requirement("pydantic"),
            requirement("pandas"),
            requirement("Pillow"),
        ],
    }),
)
