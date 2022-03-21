target "_all_platforms" {
    platforms = ["linux/amd64", "linux/arm64/v8", "linux/riscv64"]
}

function "TagWithArch" {
    params = [repo, tag, no_arch, arch]
    result = [no_arch =="1"?"${repo}:${tag}":"${repo}:${tag}-${arch}",]
}

variable "ORG" {
    default = "aarnphm"
}

variable "BENTOML_VERSION" {
	default = ""
}

variable "PYTHON_VERSION" {
	default = "3.10"
}

variable "TAG" {
    default = "1.1.0"
}

/* ------------------- */

// Special target: https://github.com/docker/metadata-action#bake-definition
target "meta-helper" {
    tags = ["${ORG}/bentoml-docker:test"]
}

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

group "test" {
	targets = ["test-runtime", "test-cudnn"]
}

target "bats-test" {
    inherits = ["_all_platforms"]
    tags = TagWithArch("${ORG}/bats-test", TAG, "1", "")
    dockerfile = "./hack/dockerfiles/bats-test.Dockerfile"
	cache-to = ["type=inline"]
	cache-from = ["${ORG}/bats-test:${TAG}"]
	target = "test"
}

target "_test-base" {
    context = "tests"
    target = "test"
	args = {
		ORGANIZATION = "${ORG}"
		PYTHON_VERSION = "${PYTHON_VERSION}"
		BENTOML_VERSION = "${BENTOML_VERSION}"
	}
}

target "_test_runtime" {
	inherits = ["_test-base", "_all_platforms"]
}

target "_test_cudnn" {
	inherits = ["_test-base", "_all_platforms"]
	args = {
        TEST_TYPE = "cudnn"
    }
}

group "test-runtime" {
    targets = ["test-runtime-ubi8", "test-runtime-alpine", "test-runtime-debian11", "test-runtime-debian10", "test-runtime-amazonlinux2"]
}

target "test-runtime-debian11" {
	inherits = ["_test_runtime"]
	target = "test-debian11"
}
target "test-runtime-debian10" {
	inherits = ["_test_runtime"]
	target = "test-debian10"
}
target "test-runtime-alpine" {
	inherits = ["_test_runtime"]
	target = "test-alpine"
}
target "test-runtime-ubi8" {
	inherits = ["_test_runtime"]
	target = "test-ubi8"
}
target "test-runtime-amazonlinux2" {
	inherits = ["_test_runtime"]
	target = "test-amazonlinux2"
}

group "test-cudnn" {
    targets = ["test-cudnn-ubi8", "test-cudnn-alpine", "test-cudnn-debian11", "test-cudnn-debian10", "test-cudnn-amazonlinux2"]
}

target "test-cudnn-debian11" {
	inherits = ["_test_cudnn"]
	target = "test-debian11"
}
target "test-cudnn-debian10" {
	inherits = ["_test_cudnn"]
	target = "test-debian10"
}
target "test-cudnn-alpine" {
	inherits = ["_test_cudnn"]
	target = "test-alpine"
}
target "test-cudnn-ubi8" {
	inherits = ["_test_cudnn"]
	target = "test-ubi8"
}
target "test-cudnn-amazonlinux2" {
	inherits = ["_test_cudnn"]
	target = "test-amazonlinux2"
}
