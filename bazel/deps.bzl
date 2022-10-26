load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

def internal_deps():
    maybe(
        http_archive,
        name = "bazel_skylib",
        sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "io_bazel_rules_go",
        sha256 = "099a9fb96a376ccbbb7d291ed4ecbdfd42f6bc822ab77ae6f1b5cb9e914e94fa",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
            "https://github.com/bazelbuild/rules_go/releases/download/v0.35.0/rules_go-v0.35.0.zip",
        ],
    )
    maybe(
        http_archive,
        name = "bazel_gazelle",
        sha256 = "efbbba6ac1a4fd342d5122cbdfdb82aeb2cf2862e35022c752eaddffada7c3f3",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v0.27.0/bazel-gazelle-v0.27.0.tar.gz",
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.27.0/bazel-gazelle-v0.27.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_proto",
        sha256 = "e017528fd1c91c5a33f15493e3a398181a9e821a804eb7ff5acdd1d2d6c2b18d",
        strip_prefix = "rules_proto-4.0.0-3.20.0",
        urls = [
            "https://github.com/bazelbuild/rules_proto/archive/refs/tags/4.0.0-3.20.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "rules_proto_grpc",
        sha256 = "507e38c8d95c7efa4f3b1c0595a8e8f139c885cb41a76cab7e20e4e67ae87731",
        strip_prefix = "rules_proto_grpc-4.1.1",
        urls = ["https://github.com/rules-proto-grpc/rules_proto_grpc/archive/4.1.1.tar.gz"],
    )

    # grpc/grpc dependencies
    # Currently c3714eced8c51db9092e0adc2a1dfb715655c795  address
    # some build issues with upb for C++.
    # TODO: Update this to v1.50.0 when 1.50.0 is out.
    maybe(
        http_archive,
        name = "com_github_grpc_grpc",
        strip_prefix = "grpc-c3714eced8c51db9092e0adc2a1dfb715655c795",
        sha256 = "3ad2b790589ca55928128513955562c4d397722afbe9b426e63a964ee59cb02a",
        urls = [
            "https://github.com/grpc/grpc/archive/c3714eced8c51db9092e0adc2a1dfb715655c795.tar.gz",
        ],
    )

    # Override the abseil-cpp version defined in grpc_deps(), which doesn't work on latest macOS
    # Fixes https://github.com/bazelbuild/bazel/issues/15168
    # This section is excerpted from https://github.com/bazelbuild/bazel/blob/master/distdir_deps.bzl
    maybe(
        http_archive,
        name = "com_google_absl",
        sha256 = "dcf71b9cba8dc0ca9940c4b316a0c796be8fab42b070bb6b7cab62b48f0e66c4",
        strip_prefix = "abseil-cpp-20211102.0",
        urls = [
            "https://mirror.bazel.build/github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/refs/tags/20211102.0.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "com_google_googleapis",
        sha256 = "5bb6b0253ccf64b53d6c7249625a7e3f6c3bc6402abd52d3778bfa48258703a0",
        strip_prefix = "googleapis-2f9af297c84c55c8b871ba4495e01ade42476c92",
        urls = [
            "https://mirror.bazel.build/github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
            "https://github.com/googleapis/googleapis/archive/2f9af297c84c55c8b871ba4495e01ade42476c92.tar.gz",
        ],
    )

    maybe(
        http_archive,
        name = "upb",
        sha256 = "03b642a535656560cd95cab3b26e8c53ce37e472307dce5bb7e47c9953bbca0f",
        strip_prefix = "upb-e5f26018368b11aab672e8e8bb76513f3620c579",
        urls = [
            "https://mirror.bazel.build/github.com/protocolbuffers/upb/archive/e5f26018368b11aab672e8e8bb76513f3620c579.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/e5f26018368b11aab672e8e8bb76513f3620c579.tar.gz",
        ],
    )

    # install buildifier
    maybe(
        http_archive,
        name = "com_github_bazelbuild_buildtools",
        sha256 = "ae34c344514e08c23e90da0e2d6cb700fcd28e80c02e23e4d5715dddcb42f7b3",
        strip_prefix = "buildtools-4.2.2",
        urls = [
            "https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz",
        ],
    )

    # python rules
    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "8c8fe44ef0a9afc256d1e75ad5f448bb59b81aba149b8958f02f7b3a98f5d9b4",
        strip_prefix = "rules_python-0.13.0",
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.13.0.tar.gz",
    )

    # io_grpc_grpc_java is for java_grpc_library and related dependencies.
    # Using commit 0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8
    maybe(
        http_archive,
        name = "io_grpc_grpc_java",
        sha256 = "35189faf484096c9eb2928c43b39f2457d1ca39046704ba8c65a69482f8ceed5",
        strip_prefix = "grpc-java-0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8",
        urls = ["https://github.com/grpc/grpc-java/archive/0cda133c52ed937f9b0a19bcbfc36bf2892c7aa8.tar.gz"],
    )
    maybe(
        http_archive,
        name = "rules_jvm_external",
        sha256 = "c21ce8b8c4ccac87c809c317def87644cdc3a9dd650c74f41698d761c95175f3",
        strip_prefix = "rules_jvm_external-1498ac6ccd3ea9cdb84afed65aa257c57abf3e0a",
        url = "https://github.com/bazelbuild/rules_jvm_external/archive/1498ac6ccd3ea9cdb84afed65aa257c57abf3e0a.zip",
    )

    # loading kotlin rules
    # first to load grpc/grpc-kotlin
    maybe(
        http_archive,
        name = "com_github_grpc_grpc_kotlin",
        sha256 = "b1ec1caa5d81f4fa4dca0662f8112711c82d7db6ba89c928ca7baa4de50afbb2",
        strip_prefix = "grpc-kotlin-a1659c1b3fb665e01a6854224c7fdcafc8e54d56",
        urls = ["https://github.com/grpc/grpc-kotlin/archive/a1659c1b3fb665e01a6854224c7fdcafc8e54d56.tar.gz"],
    )
    maybe(
        http_archive,
        name = "io_bazel_rules_kotlin",
        sha256 = "a57591404423a52bd6b18ebba7979e8cd2243534736c5c94d35c89718ea38f94",
        urls = ["https://github.com/bazelbuild/rules_kotlin/releases/download/v1.6.0/rules_kotlin_release.tgz"],
    )
