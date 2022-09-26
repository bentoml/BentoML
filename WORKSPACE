workspace(name = "bentoml")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# install buildifier
http_archive(
    name = "com_github_bazelbuild_buildtools",
    sha256 = "ae34c344514e08c23e90da0e2d6cb700fcd28e80c02e23e4d5715dddcb42f7b3",
    strip_prefix = "buildtools-4.2.2",
    urls = [
        "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
    ],
)

# setup rules_proto and rules_proto_grpc
http_archive(
    name = "rules_proto",
    sha256 = "e017528fd1c91c5a33f15493e3a398181a9e821a804eb7ff5acdd1d2d6c2b18d",
    strip_prefix = "rules_proto-4.0.0-3.20.0",
    urls = [
        "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0-3.20.0.tar.gz",
    ],
)

http_archive(
    name = "rules_proto_grpc",
    sha256 = "507e38c8d95c7efa4f3b1c0595a8e8f139c885cb41a76cab7e20e4e67ae87731",
    strip_prefix = "rules_proto_grpc-4.1.1",
    urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.1.1.tar.gz"],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")
load("@rules_proto_grpc//:repositories.bzl", "rules_proto_grpc_repos", "rules_proto_grpc_toolchains")

rules_proto_grpc_toolchains()

rules_proto_grpc_repos()

rules_proto_dependencies()

rules_proto_toolchains()

# We need to load go_grpc rules first
load("@rules_proto_grpc//:repositories.bzl", "bazel_gazelle", "io_bazel_rules_go")  # buildifier: disable=same-origin-load

io_bazel_rules_go()

bazel_gazelle()

load("@rules_proto_grpc//go:repositories.bzl", rules_proto_grpc_go_repos = "go_repos")

rules_proto_grpc_go_repos()

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.19")

# grpc/grpc dependencies
# Currently c3714eced8c51db9092e0adc2a1dfb715655c795  address
# some build issues with upb for C++.
# TODO: Update this to v1.50.0 when 1.50.0 is out.
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-3.19.4",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.19.4.zip"],
)
http_archive(
    name = "com_github_grpc_grpc",
    strip_prefix = "grpc-c3714eced8c51db9092e0adc2a1dfb715655c795",
    urls = [
        "https://github.com/grpc/grpc/archive/c3714eced8c51db9092e0adc2a1dfb715655c795.tar.gz",
    ],
)

# Override the abseil-cpp version defined in grpc_deps(), which doesn't work on latest macOS
# Fixes https://github.com/bazelbuild/bazel/issues/15168
# This section is excerpted from https://github.com/bazelbuild/bazel/blob/master/distdir_deps.bzl
http_archive(
    name = "com_google_absl",
    sha256 = "dcf71b9cba8dc0ca9940c4b316a0c796be8fab42b070bb6b7cab62b48f0e66c4",
    strip_prefix = "abseil-cpp-20211102.0",
    urls = [
        "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
        "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
    ],
)

# Projects using gRPC as an external dependency must call both grpc_deps() and
# grpc_extra_deps().
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

http_archive(
    name = "com_google_googleapis",
    sha256 = "5bb6b0253ccf64b53d6c7249625a7e3f6c3bc6402abd52d3778bfa48258703a0",
    strip_prefix = "googleapis-2f9af297c84c55c8b871ba4495e01ade42476c92",
    urls = [
        "https://mirror.bazel.build/github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
        "https://github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
    ],
)

http_archive(
    name = "upb",
    sha256 = "03b642a535656560cd95cab3b26e8c53ce37e472307dce5bb7e47c9953bbca0f",
    strip_prefix = "upb-e5f26018368b11aab672e8e8bb76513f3620c579",
    urls = [
        "https://mirror.bazel.build/github.com/protocolbuffers/upb/archive/e5f26018368b11aab672e8e8bb76513f3620c579.tar.gz",
        "https://github.com/protocolbuffers/upb/archive/e5f26018368b11aab672e8e8bb76513f3620c579.tar.gz",
    ],
)

http_archive(
    name = "bazel_gazelle",
    sha256 = "de69a09dc70417580aabf20a28619bb3ef60d038470c7cf8442fafcf627c21cb",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.24.0/bazel-gazelle-v0.24.0.tar.gz",
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")

gazelle_dependencies()

# load python rules here
# Using commit f0efec5cf8c0ae16483ee677a09ec70737a01bf5
http_archive(
    name = "rules_python",
    strip_prefix = "rules_python-f0efec5cf8c0ae16483ee677a09ec70737a01bf5",
    url = "https://github.com/bazelbuild/rules_python/archive/f0efec5cf8c0ae16483ee677a09ec70737a01bf5.zip",
)

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "bentoml_requirements",
    requirements_lock = "//grpc-client/python:requirements.lock.txt",
)

# Load the starlark macro which will define your dependencies.
load("@bentoml_requirements//:requirements.bzl", "install_deps")

# Call it to define repos for your requirements.
install_deps()

# io_grpc_grpc_java is for java_grpc_library and related dependencies.
# Using commit 0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8
http_archive(
    name = "io_grpc_grpc_java",
    sha256 = "35189faf484096c9eb2928c43b39f2457d1ca39046704ba8c65a69482f8ceed5",
    strip_prefix = "grpc-java-0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8",
    urls = ["https://github.com/grpc/grpc-java/archive/0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8.tar.gz"],
)

http_archive(
    name = "rules_jvm_external",
    sha256 = "c21ce8b8c4ccac87c809c317def87644cdc3a9dd650c74f41698d761c95175f3",
    strip_prefix = "rules_jvm_external-1498ac6ccd3ea9cdb84afed65aa257c57abf3e0a",
    url = "https://github.com/bazelbuild/rules_jvm_external/archive/1498ac6ccd3ea9cdb84afed65aa257c57abf3e0a.zip",
)

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

# loading kotlin rules
# first to load grpc/grpc-kotlin
http_archive(
    name = "com_github_grpc_grpc_kotlin",
    sha256 = "b1ec1caa5d81f4fa4dca0662f8112711c82d7db6ba89c928ca7baa4de50afbb2",
    strip_prefix = "grpc-kotlin-a1659c1b3fb665e01a6854224c7fdcafc8e54d56",
    urls = ["https://github.com/grpc/grpc-kotlin/archive/a1659c1b3fb665e01a6854224c7fdcafc8e54d56.tar.gz"],
)

http_archive(
    name = "io_bazel_rules_kotlin",
    sha256 = "a57591404423a52bd6b18ebba7979e8cd2243534736c5c94d35c89718ea38f94",
    urls = ["https://github.com/bazelbuild/rules_kotlin/releases/download/v1.6.0/rules_kotlin_release.tgz"],
)

load("@io_bazel_rules_kotlin//kotlin:repositories.bzl", "kotlin_repositories")

kotlin_repositories()

load("@io_bazel_rules_kotlin//kotlin:core.bzl", "kt_register_toolchains")

kt_register_toolchains()

# swift rules
# TODO: Currently fails at detecting compiled gRPC swift library
# Since CgRPC is deprecated, seems like no rules are being maintained
# for the newer swift implementation.
