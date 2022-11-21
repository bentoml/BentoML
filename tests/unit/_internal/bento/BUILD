load("//rules:python.bzl", "py_test")

filegroup(
    name = "test_bentos",
    srcs = glob(["bento*/**/*"]) + glob(["simplebento/**/*"]),
)

py_test(
    name = "bento",
    data = [
        ":test_bentos",
        "//tests/unit:conftest",
    ],
    imports = ["./"],
)
