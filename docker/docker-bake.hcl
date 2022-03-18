target "_all_platforms" {
    platforms = ["linux/amd64", "linux/arm64/v8", "linux/arm/v7", "linux/arm/v6", "linux/arm/v5","linux/ppc64le", "linux/s390x", "linux/riscv64", "linux/mips64le"]
}

function "TagWithArch" {
    params = [repo, tag, no_arch, arch]
    result = [no_arch =="1"?"${repo}:${tag}":"${repo}:${tag}-${arch}",]
}

/* ------------------- */
variable "MANAGER_REPO" {
    default = "aarnphm/bentoml-docker"
}
variable "MANAGER_TAG"{
    default = "1.1.0"
}

target "all" {
    platforms = ["linux/amd64", "linux/arm64/v8"]
    inherits = ["_all_platforms"]
    tags = TagWithArch(MANAGER_REPO, MANAGER_TAG, "1", "")
    dockerfile = "./hack/dockerfiles/dev.Dockerfile"
	cache-to = ["type=inline"]
	pull = true
	cache-from = ["${MANAGER_REPO}"]
}

/* ------------------- */
variable "TEST_REPO" {
    default = "aarnphm/bats-test"
}

target "test" {
    inherits = ["_all_platforms"]
    tags = TagWithArch(TEST_REPO, "latest", "1", "")
    dockerfile = "./hack/dockerfiles/test.Dockerfile"
	cache-to = ["type=inline"]
	cache-from = ["${TEST_REPO}"]
	target = "test"
}

target "test-platforms" {
    inherits = ["_all_platforms"]
    dockerfile = "./hack/dockerfiles/platform.Dockerfile"
}
