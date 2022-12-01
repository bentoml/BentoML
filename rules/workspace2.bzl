load("@rules_nodejs//nodejs:repositories.bzl", "nodejs_register_toolchains")

# Install the yarn tool
load("@rules_nodejs//nodejs:yarn_repositories.bzl", "yarn_repositories")
load("@build_bazel_rules_nodejs//:index.bzl", "yarn_install")

# NOTE: grpc_extra_deps register toolchain to 1.18
# therefore, we need to load this afterwards.
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

def _bentoml_workspace():
    grpc_extra_deps()

    nodejs_register_toolchains(
        name = "nodejs",
        node_version = "16.16.0",
    )

    yarn_repositories(
        name = "yarn",
        node_repository = "nodejs",
    )

    yarn_install(
        name = "npm",
        exports_directories_only = False,  # Required for ts_library
        package_json = "//:package.json",
        package_path = "/",
        yarn = "@yarn//:bin/yarn",
        yarn_lock = "//:yarn.lock",
    )

workspace2 = _bentoml_workspace
