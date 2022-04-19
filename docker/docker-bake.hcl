target "_all_platforms" {
    platforms = ["linux/amd64", "linux/arm64/v8"]
}

function "TagWithArch" {
    params = [repo, tag, no_arch, arch]
    result = [no_arch =="1"?"${repo}:${tag}":"${repo}:${tag}-${arch}",]
}

variable "ORG" {
    default = "aarnphm"
}

variable "TAG" {
    default = "1.1.0"
}

/* ------------------- */

// Special target: https://github.com/docker/metadata-action#bake-definition
target "meta-helper" {
    tags = ["${ORG}/bentoml-docker:bats-test"]
}

target "shared" {
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
	targets = ["_test_runtime", "_test_cudnn"]
}

target "bats-test" {
    inherits = ["_all_platforms"]
    tags = TagWithArch("${ORG}/bats-test", TAG, "1", "")
	cache-to = ["type=inline"]
	cache-from = ["${ORG}/bats-test:${TAG}"]
    context = "tests"
	target = "bats-test"
}

target "_test-base" {
    context = "tests"
    target = "test"
	args = {
        TEST_TYPE = "runtime"
	}
}

target "_test_runtime" {
	inherits = ["_test-base", "_all_platforms"]
    tags = TagWithArch("${ORG}/bentoml-docker", "test-runtime", "1", "")
	cache-to = ["type=inline"]
	cache-from = ["${ORG}/bentoml-docker:test-runtime"]
}

target "_test_cudnn" {
	inherits = ["_test-base", "_all_platforms"]
    tags = TagWithArch("${ORG}/bentoml-docker", "test-cudnn", "1", "")
	cache-to = ["type=inline"]
	cache-from = ["${ORG}/bentoml-docker:test-cudnn"]
	args = {
        TEST_TYPE = "cudnn"
    }
}
