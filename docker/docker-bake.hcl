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

variable "ORG" {
    default = "aarnphm"
}

/* ------------------- */

target "shared" {
    platforms = ["linux/amd64", "linux/arm64/v8"]
    inherits = ["_all_platforms"]
	cache-to = ["type=inline"]
    dockerfile = "./hack/dockerfiles/builder.Dockerfile"
	pull = true
    context = "."
}

target "manager" {
    inherits = ["shared"]
    tags = TagWithArch("${ORG}/bentoml-docker", TAG, "1", "")
    target = "releases"
	cache-from = ["${ORG}/bentoml-docker:${TAG}"]
}

target "base" {
    inherits = ["shared"]
    tags = TagWithArch("${ORG}/bentoml-docker", "build", "1", "")
    target = "build"
	cache-from = ["${ORG}/bentoml-docker:build"]
}

group default {
    targets = ["manager"]
}

/* ------------------- */

target "test" {
    inherits = ["_all_platforms"]
    tags = TagWithArch("${ORG}/bats-test", TAG, "1", "")
    dockerfile = "./hack/dockerfiles/bats-test.Dockerfile"
	cache-to = ["type=inline"]
	cache-from = ["${ORG}/bats-test:${TAG}"]
	target = "test"
}
