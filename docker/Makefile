GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
UNAME_M ?= $(shell uname -m)

BENTOML_VERSION := $(shell git describe --tags `git rev-list --tags --max-count=1` | sed 's/v\(\)/\1/; s/-//g')

GENERATE_ARGS := --bentoml-version $(BENTOML_VERSION) --overwrite
BUILD_TESTS_ARGS := --max-workers 5 --bentoml-version $(BENTOML_VERSION)
IMAGE_NAME := bento-server

# Some Dockerfiles use features that are only supported with BuildKit enabled
export DOCKER_BUILDKIT=1

# Create multi-platform docker image. If you have native systems around, using
# them will be much more efficient at build time. See e.g.
BUILDXDETECT = ${HOME}/.docker/cli-plugins/docker-buildx
# Just one of the many files created
QEMUDETECT = /proc/sys/fs/binfmt_misc/qemu-aarch64

MANAGER_IMAGE := bentoml-docker:1.0
RUN_IMAGE := ${MANAGER_IMAGE}-$(shell uname -m)

DOCKER_ARGS := -u 1034:1034 \
	--privileged --rm --init --net=host \
	-v $(realpath /etc/localtime):/etc/localtime:ro \
	-v ${PWD}:/bentoml \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-w /bentoml ${RUN_IMAGE}

DOCKER_RUN := docker run ${DOCKER_ARGS}

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Format
	@poetry run python -m autoflake --in-place --remove-all-unused-imports manager/*.py
	@${GIT_ROOT}/scripts/tools/formatter.sh

lint: ## Lint
	@pylint --rcfile="${GIT_ROOT}/pyproject.toml" --disable=E0401,F0010 .

.PHONY: install-local
install-local: ## Install manager locally
	@pip install -e .

run-generate: ## Generate all required files
	manager generate $(GENERATE_ARGS)

run-build: ## Build all releases images
	manager build $(BUILD_TESTS_ARGS)

run-test: ${IMAGE_NAME} ## Run tests
${IMAGE_NAME}:
	@echo "Running sanity check for ${IMAGE_NAME} on all releases."
	manager run-tests ${BUILD_TESTS_ARGS}


qemu: ${QEMUDETECT}
${QEMUDETECT}:
	docker run --privileged multiarch/qemu-user-static --reset -p yes
	docker ps -a | sed -n 's, *multiarch/qemu-user-static.*,,p' \
	  | (xargs docker rm 2>&1 || \
	    echo "Cannot remove docker container on ZFS; retry after next reboot") \
	  | grep -v 'dataset is busy'

buildx: ${BUILDXDETECT}
${BUILDXDETECT}:
	@if ! `docker buildx &>/dev/null`; then \
		echo; \
		echo '*** `docker buildx` missing. Install binary for this machine architecture'; \
		echo '*** from `https://github.com/docker/buildx/releases/latest`'; \
		echo '*** to `~/.docker/cli-plugins/docker-buildx` and `chmod +x` it.'; \
		echo; \
		exit 1; \
	fi

emulator:  buildx ## Install all emulator
	@docker run --privileged --rm tonistiigi/binfmt --install all
	@echo 'Finish installed emulator from tonistiig/binfmt.'

platform-check: emulator ## Test supported platform string
	@docker buildx ls | grep platform_check_builder &>/dev/null | docker buildx use platform_check_builder || docker buildx create --use --driver docker-container --name platform_check_builder;
	@docker buildx bake test-platforms

docker-build: ${UNAME_M}

.PHONY: ${UNAME_M}
${UNAME_M}:
	$(eval bake_target := build-$@)
	 docker buildx bake $(bake_target)

docker-run-generate: ## Run generate from a docker container (useful for CI)
	$(DOCKER_RUN) poetry run manager generate $(GENERATE_ARGS)

docker-run-build: ## Run build from a docker container (useful for CI)
	$(DOCKER_RUN) poetry run manager build $(BUILD_TESTS_ARGS)
