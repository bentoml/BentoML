load("@io_bazel_rules_kotlin//kotlin:jvm.bzl", "kt_jvm_binary")
load("@io_grpc_grpc_java//:java_grpc_library.bzl", "java_grpc_library")
load("@com_github_grpc_grpc_kotlin//:kt_jvm_grpc.bzl", "kt_jvm_grpc_library", "kt_jvm_proto_library")

java_proto_library(
    name = "service_java",
    deps = ["//:service_v1_proto"],
)

kt_jvm_proto_library(
    name = "service_kt",
    deps = ["//:service_v1_proto"],
)

kt_jvm_grpc_library(
    name = "service_grpc_kt",
    srcs = ["//:service_v1_proto"],
    deps = [":service_java"],
)

kt_jvm_binary(
    name = "client_kt",
    srcs = ["src/main/kotlin/com/client/BentoServiceClient.kt"],
    main_class = "com.client.BentoServiceClient",
    deps = [
        ":service_grpc_kt",
        ":service_kt",
        "@com_google_protobuf//:protobuf_java_util",
        "@io_grpc_grpc_java//netty",
    ],
)
