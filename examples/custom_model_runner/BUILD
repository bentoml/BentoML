load("//rules:utils.bzl", "run_shell")

package(default_visibility = ["//visibility:private"])

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
