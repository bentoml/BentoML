load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

# NOTE: the two following macros set up rules_go and bazel_gazelle
load("@rules_proto_grpc//:repositories.bzl", "bazel_gazelle", "io_bazel_rules_go")  # buildifier: disable=same-origin-load
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")
load("@build_bazel_rules_nodejs//:repositories.bzl", "build_bazel_rules_nodejs_dependencies")
load("@maven//:compat.bzl", "compat_repositories")
load("@io_bazel_rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

def _bentoml_workspace():
    protobuf_deps()
    grpc_deps()

    io_bazel_rules_go()

    bazel_gazelle()

    gazelle_dependencies(go_sdk = "go_sdk")

    compat_repositories()
    kt_register_toolchains()

    build_bazel_rules_nodejs_dependencies()

workspace1 = _bentoml_workspace
