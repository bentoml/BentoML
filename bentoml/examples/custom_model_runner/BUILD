package(default_visibility = ["//visibility:private"])

load("//bazel:utils.bzl", "run_shell")

run_shell(
    name = "download_mnist_data",
    content = [
        " | ".join([
            "wget -qO- https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz",
            "tar xz",
        ]),
    ],
)

run_shell(
    name = "train",
    content = ["python train.py $@"],
)
