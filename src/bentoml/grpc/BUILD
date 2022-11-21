package(default_visibility = ["//src/bentoml:__subpackages__"])

exports_files(["buf.yaml"])

filegroup(
    name = "srcs_files",
    srcs = glob(["**/*.py"]) + [
        "//src/bentoml/grpc/v1alpha1:srcs_files",
        "//src/bentoml/grpc/v1:srcs_files",
    ],
)
