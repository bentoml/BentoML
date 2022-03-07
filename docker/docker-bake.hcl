variable "IMAGE_NAME" {
    default = "bentoml-docker"
}
variable "TAG" {
    default = "1.0"
}
variable "FULL_NAME" {
    default = "${IMAGE_NAME}:${TAG}"
}
variable "USE_GLIBC" {
    default = ""
}

target "platforms" {
    platforms = concat(["linux/amd64", "linux/386", "linux/arm64/v8", "linux/arm/v7","linux/arm/v6", "linux/ppc64le", "linux/s390x", "linux/riscv64","linux/mips64le","darwin/amd64", "darwin/arm64", "windows/amd64", "windows/arm", "windows/386"], USE_GLIBC!=""?[]:["windows/arm64"])
}

function "tag_arch" {
    params = [arch]
    result = "${FULL_NAME}-${arch}"
}

target "build-arm64" {
    platforms = ["linux/arm64/v8"]
    dockerfile = "./hack/dev.Dockerfile"
    tags = [tag_arch("arm64"),]
    output = ["type=docker"]
}

target "build-amd64" {
    platforms = ["linux/amd64"]
    dockerfile = "./hack/dev.Dockerfile"
    tags = [tag_arch("amd64"),]
    output = ["type=docker"]
}

target "test-platforms" {
    inherits = ["platforms"]
    dockerfile = "./hack/platform.Dockerfile"
}