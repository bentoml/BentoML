package(default_visibility = ["//visibility:public"])

LATEST_VERSION = "v1"

[
    alias(
        name = lang + "_" + version,
        actual = "//grpc-client/{}:client_{}".format(lang, version),
        visibility = ["//visibility:public"],
    )
    for lang in [
        "cpp",
        "go",
        "java",
        "kotlin",
        "node",
        "python",
        "swift",
    ]
    for version in [
        "v1alpha1",
        "v1",
    ]
]

[
    alias(
        name = lang,
        actual = "//grpc-client/{}:client_{}".format(lang, LATEST_VERSION),
        visibility = ["//visibility:public"],
    )
    for lang in [
        "cpp",
        "go",
        "java",
        "kotlin",
        "node",
        "python",
        "swift",
    ]
]
