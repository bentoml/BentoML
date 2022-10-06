# TODO: Migrate to bzlmod once 6.0.0 is released.
workspace(name = "com_github_bentoml_bentoml")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("//rules:internal.bzl", internal_deps = "bentoml_internal_deps")

internal_deps()

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")
load("@bazel_skylib//lib:unittest.bzl", "register_unittest_toolchains")

bazel_skylib_workspace()

register_unittest_toolchains()

# setup rules_proto_grpc
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")

rules_proto_grpc_toolchains()

rules_proto_grpc_repos()

rules_proto_dependencies()

rules_proto_toolchains()

# NOTE: the two following macros set up rules_go and bazel_gazelle
load("@rules_proto_grpc//:repositories.bzl", "bazel_gazelle", "io_bazel_rules_go")  # buildifier: disable=same-origin-load

io_bazel_rules_go()

bazel_gazelle()

load("@rules_proto_grpc//go:repositories.bzl", rules_proto_grpc_go_repos = "go_repos")

rules_proto_grpc_go_repos()

load("@rules_proto_grpc//php:repositories.bzl", rules_proto_grpc_php_repos = "php_repos")

rules_proto_grpc_php_repos()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# NOTE: grpc_extra_deps register toolchain to 1.18
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

# load go tooling, buf
load("@rules_buf//buf:repositories.bzl", "rules_buf_dependencies", "rules_buf_toolchains")

rules_buf_dependencies()

rules_buf_toolchains(version = "v1.9.0")

# NOTE: rules_python
load("@rules_python//python:pip.bzl", "pip_parse")
load("@rules_python//python/pip_install:pip_repository.bzl", "pip_repository")

pip_parse(
    name = "pypi",
    requirements_lock = "//requirements/bazel:pypi.lock.txt",
)

pip_parse(
    name = "frameworks",
    requirements_darwin = "//requirements/bazel:frameworks-macos.lock.txt",
    requirements_linux = "//requirements/bazel:frameworks-linux.lock.txt",
    requirements_windows = "//requirements/bazel:frameworks-windows.lock.txt",
)

load("@pypi//:requirements.bzl", pypi_deps = "install_deps")

pypi_deps()

load("@frameworks//:requirements.bzl", framework_deps = "install_deps")

framework_deps()

load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies")

aspect_bazel_lib_dependencies()

# NOTE: Java and Kotlin setup.
load("@rules_jvm_external//:defs.bzl", "maven_install")
load("@io_grpc_grpc_java//:repositories.bzl", "IO_GRPC_GRPC_JAVA_ARTIFACTS", "IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS", "grpc_java_repositories")

IO_GRPC_GRPC_KOTLIN_ARTIFACTS = [
    "com.squareup:kotlinpoet:1.11.0",
    "org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.2",
    "org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.6.2",
    "org.jetbrains.kotlinx:kotlinx-coroutines-debug:1.6.2",
]

maven_install(
    artifacts = [
        "com.google.jimfs:jimfs:1.1",
        "com.google.truth.extensions:truth-proto-extension:1.0.1",
        "com.google.protobuf:protobuf-kotlin:3.18.0",
    ] + IO_GRPC_GRPC_KOTLIN_ARTIFACTS + IO_GRPC_GRPC_JAVA_ARTIFACTS,
    generate_compat_repositories = True,
    override_targets = IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS,
    repositories = [
        "https://repo.maven.apache.org/maven2/",
    ],
)

load("@maven//:compat.bzl", "compat_repositories")

compat_repositories()

grpc_java_repositories()

load("@io_bazel_rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()

load("@io_bazel_rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()

# TODO: setup container rules and utilities to run bazel in docker.
load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load("@io_bazel_rules_docker//container:pull.bzl", "container_pull")

container_pull(
    name = "python3_slim_amd64",
    digest = "sha256:07d8280c273cb45f1f6dbbe06578681eb7a8937e1224b1182b98080b01a41d01",
    registry = "index.docker.io",
    repository = "library/python",
    tag = "3.7-slim",
)

http_file(
    name = "bazel_gpg",
    sha256 = "8375bd5de1778a9fbb58a482a7ce9444ab9b1f6bb5fddd3700ae86b3fe0e4d3a",
    urls = ["https://bazel.build/bazel-release.pub.gpg"],
)

# NOTE: rules_nodejs
load("@rules_proto_grpc//js:repositories.bzl", rules_proto_grpc_js_repos = "js_repos")

rules_proto_grpc_js_repos()

load("@build_bazel_rules_nodejs//:repositories.bzl", "build_bazel_rules_nodejs_dependencies")

build_bazel_rules_nodejs_dependencies()

load("@rules_nodejs//nodejs:repositories.bzl", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = "16.16.0",
)

# Install the yarn tool
load("@rules_nodejs//nodejs:yarn_repositories.bzl", "yarn_repositories")

yarn_repositories(
    name = "yarn",
    node_repository = "nodejs",
)

load("@build_bazel_rules_nodejs//:index.bzl", "yarn_install")

yarn_install(
    name = "npm",
    exports_directories_only = False,  # Required for ts_library
    package_json = "//:package.json",
    package_path = "/",
    symlink_node_modules = True,
    yarn = "@yarn//:bin/yarn",
    yarn_lock = "//:yarn.lock",
)

# NOTE: rules_swift and rules_apple
load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)

apple_support_dependencies()

load(
    "@build_bazel_rules_swift//swift:repositories.bzl",
    "swift_rules_dependencies",
)

swift_rules_dependencies()

load(
    "@build_bazel_rules_swift//swift:extras.bzl",
    "swift_rules_extra_dependencies",
)

swift_rules_extra_dependencies()

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()
