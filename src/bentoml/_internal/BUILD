package(default_visibility = ["//src/bentoml:__pkg__"])

filegroup(
    name = "srcs_files",
    srcs = glob(["**/*.py"]),
)

filegroup(
    name = "data_files",
    srcs = glob(["configuration/**/*.yaml"]) + glob(["server/static_content/**"]) + glob(["container/frontend/**"]),
)
