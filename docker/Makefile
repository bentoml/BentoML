GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

VERSIONTAGS = $(shell sed -n -e 's,^VERSION = .\(\([0-9].[0-9]\).[0-9]\).$$,-v :\1 -v :\2,p' manager/__version__.py)
PLATFORM_CHECK  = linux/amd64,linux/arm64,linux/arm/v7,linux/arm/v6,linux/ppc64le,linux/386,linux/riscv64,linux/mips64le
BENTOML_VERSION := $(shell git describe --tags `git rev-list --tags --max-count=1` | sed 's/v\(\)/\1/; s/-//g')

# Create multi-platform docker image. If you have native systems around, using
# them will be much more efficient at build time. See e.g.
BUILDXDETECT = ${HOME}/.docker/cli-plugins/docker-buildx
# Just one of the many files created
QEMUDETECT = /proc/sys/fs/binfmt_misc/qemu-m68k

DOCKER_IMG := bentoml-docker:1.0

DOCKER_ARGS := -u $(shell id -u):$(shell id -g) \
	--privileged --rm --init --net=host \
	-v $(realpath /etc/localtime):/etc/localtime:ro \
	-v ${PWD}:/bentoml \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-w /bentoml ${DOCKER_IMG}-$(shell uname -m)

DOCKER_BUILD_PLATFORM := linux/amd64 linux/arm64/v8 
DOCKER_BUILD_ARGS := --load -f ${PWD}/Dockerfile

DOCKER_RUN_MANAGER := docker run ${DOCKER_ARGS} poetry run manager

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Format
	@poetry run python -m autoflake --in-place --remove-all-unused-imports manager/**.py
	@${GIT_ROOT}/scripts/tools/formatter.sh

lint: ## Lint
	@pylint --rcfile="${GIT_ROOT}/pyproject.toml" --disable=E0401,F0010 .

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
	@docker buildx ls | grep platform_check_builder &>/dev/null || docker buildx create --use --driver docker-container --name platform_check_builder;
	@docker buildx use platform_check_builder;
	@docker buildx build -f tools/Dockerfile-platform --no-cache --progress=plain --platform=${PLATFORM_CHECK} .

.PHONY: install-local
install-local: ## Install manager locally
	@pip install -e .

GENERATE_ARGS := --dump-metadata --bentoml-version $(BENTOML_VERSION)
BUILD_ARGS := --max-workers 5 --bentoml-version $(BENTOML_VERSION)

run-generate: ## Generate all required files
	manager generate $(GENERATE_ARGS)

run-build: ## Build all releases images
	manager build $(BUILD_ARGS)

docker-build:
	@echo "Building ${DOCKER_IMG} for $(DOCKER_BUILD_PLATFORM)..."
	@$(foreach platform, $(DOCKER_BUILD_PLATFORM), \
		DOCKER_BUILDKIT=1 docker buildx build $(DOCKER_BUILD_ARGS) \
			--platform $(platform) -t ${DOCKER_IMG}-$(word 2,$(subst /, ,$(platform))) ${PWD}; \
	)

docker-run-generate: ## Run generate from a docker container (useful for CI)
	$(DOCKER_RUN_MANAGER) generate $(GENERATE_ARGS)

docker-run-build: ## Run build from a docker container (useful for CI)
	$(DOCKER_RUN_MANAGER) build $(BUILD_ARGS)
