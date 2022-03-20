target "_all_platforms" {
    platforms = ["linux/amd64", "linux/arm64/v8", "linux/riscv64"]
}

function "TagWithArch" {
    params = [repo, tag, no_arch, arch]
    result = [no_arch =="1"?"${repo}:${tag}":"${repo}:${tag}-${arch}",]
}

variable "TAG"{
    default = "1.1.0"
}

/* ------------------- */
variable "MANAGER_REPO" {
    default = "aarnphm/bentoml-docker"
}

target "manager-shared" {
    platforms = ["linux/amd64", "linux/arm64/v8"]
    inherits = ["_all_platforms"]
	cache-to = ["type=inline"]
    dockerfile = "./hack/dockerfiles/dev.Dockerfile"
	pull = true
    context = "."
}

target "manager" {
    inherits = ["manager-shared"]
    tags = TagWithArch(MANAGER_REPO, TAG, "1", "")
    target = "base"
	cache-from = ["${MANAGER_REPO}:${TAG}"]
}

target "base_dev" {
    inherits = ["manager-shared"]
    tags = TagWithArch(MANAGER_REPO, "base_dev", "1", "")
    target = "base_build"
	cache-from = ["${MANAGER_REPO}:base_dev"]
}

group default {
    targets = ["manager"]
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
	cache-from = ["${TEST_REPO}:${TAG}"]
	target = "test"
}

target "test-platforms" {
    inherits = ["_all_platforms"]
    dockerfile = "./hack/dockerfiles/platform.Dockerfile"
}
