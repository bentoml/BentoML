load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

# setup rules_proto_grpc
load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")

# NOTES: rules_go
load("@rules_proto_grpc//go:repositories.bzl", rules_proto_grpc_go_repos = "go_repos")

# NOTES: rules_php
load("@rules_proto_grpc//php:repositories.bzl", rules_proto_grpc_php_repos = "php_repos")

# NOTE: rules_nodejs
load("@rules_proto_grpc//js:repositories.bzl", rules_proto_grpc_js_repos = "js_repos")

# load go tooling, buf
load("@rules_buf//buf:repositories.bzl", "rules_buf_dependencies", "rules_buf_toolchains")
load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies")

# NOTE: Java and Kotlin setup.
load("@rules_jvm_external//:defs.bzl", "maven_install")
load("@io_grpc_grpc_java//:repositories.bzl", "IO_GRPC_GRPC_JAVA_ARTIFACTS", "IO_GRPC_GRPC_JAVA_OVERRIDE_TARGETS", "grpc_java_repositories")
load("@io_bazel_rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

# TODO: setup container rules and utilities to run bazel in docker.
load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

# NOTE: rules_swift and rules_apple
load(
    "@build_bazel_apple_support//lib:repositories.bzl",
    "apple_support_dependencies",
)
load(
    "@build_bazel_rules_swift//swift:repositories.bzl",
    "swift_rules_dependencies",
)
load(
    "@build_bazel_rules_swift//swift:extras.bzl",
    "swift_rules_extra_dependencies",
)
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

IO_GRPC_GRPC_KOTLIN_ARTIFACTS = [
    "com.squareup:kotlinpoet:1.11.0",
    "org.jetbrains.kotlinx:kotlinx-coroutines-core:1.6.2",
    "org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.6.2",
    "org.jetbrains.kotlinx:kotlinx-coroutines-debug:1.6.2",
]

def _bentoml_workspace():
    bazel_skylib_workspace()

    rules_proto_grpc_toolchains()

    rules_proto_grpc_repos()

    rules_proto_dependencies()

    rules_proto_toolchains()

    rules_proto_grpc_go_repos()

    rules_proto_grpc_php_repos()

    rules_buf_dependencies()

    rules_buf_toolchains(version = "v1.9.0")

    aspect_bazel_lib_dependencies()

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

    grpc_java_repositories()

    kotlin_repositories()

    container_repositories()

    http_file(
        name = "bazel_gpg",
        sha256 = "8375bd5de1778a9fbb58a482a7ce9444ab9b1f6bb5fddd3700ae86b3fe0e4d3a",
        urls = ["https://bazel.build/bazel-release.pub.gpg"],
    )

    rules_proto_grpc_js_repos()

    apple_support_dependencies()

    swift_rules_dependencies()

    swift_rules_extra_dependencies()

    rules_foreign_cc_dependencies()

workspace0 = _bentoml_workspace
