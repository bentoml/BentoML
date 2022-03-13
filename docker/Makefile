#  ██████╗ ██████╗ ███╗   ██╗███████╗████████╗ █████╗ ███╗   ██╗████████╗███████╗
# ██╔════╝██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔══██╗████╗  ██║╚══██╔══╝██╔════╝
# ██║     ██║   ██║██╔██╗ ██║███████╗   ██║   ███████║██╔██╗ ██║   ██║   ███████╗
# ██║     ██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══██║██║╚██╗██║   ██║   ╚════██║
# ╚██████╗╚██████╔╝██║ ╚████║███████║   ██║   ██║  ██║██║ ╚████║   ██║   ███████║
#  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝

UNAME_M ?= $(shell uname -m)
GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
BENTOML_VERSION := $(shell git describe --tags `git rev-list --tags --max-count=1` | sed 's/v\(\)/\1/; s/-//g')

MANAGER_REPO := bentoml-docker
MANAGER_TAG := 1.1.0

# ██████╗  ██████╗  ██████╗██╗  ██╗███████╗██████╗
# ██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██╔════╝██╔══██╗
# ██║  ██║██║   ██║██║     █████╔╝ █████╗  ██████╔╝
# ██║  ██║██║   ██║██║     ██╔═██╗ ██╔══╝  ██╔══██╗
# ██████╔╝╚██████╔╝╚██████╗██║  ██╗███████╗██║  ██║
# ╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝

# Some Dockerfiles use features that are only supported with BuildKit enabled
export DOCKER_BUILDKIT=1

# Create multi-platform docker image. If you have native systems around, using
# them will be much more efficient at build time. See e.g.
BUILDXDETECT = ${HOME}/.docker/cli-plugins/docker-buildx
# Just one of the many files created
QEMUDETECT = /proc/sys/fs/binfmt_misc/qemu-aarch64

FULL_TAG := ${MANAGER_REPO}:${MANAGER_TAG}-${UNAME_M}

DOCKER_ARGS := -u 1034:1034 \
	--privileged --rm --init --net=host \
	-v $(realpath /etc/localtime):/etc/localtime:ro \
	-v ${PWD}:/bentoml \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-w /bentoml ${FULL_TAG}
DOCKER_RUN := docker run ${DOCKER_ARGS}

#  ██████╗ ███████╗███╗   ██╗███████╗██████╗  █████╗ ██╗
# ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔══██╗██╔══██╗██║
# ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ██████╔╝███████║██║
# ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ██╔══██╗██╔══██║██║
# ╚██████╔╝███████╗██║ ╚████║███████╗██║  ██║██║  ██║███████╗
#  ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝

.DEFAULT_GOAL := help

.PHONY: install-local
install-local: ## Install manager locally
	@pip install -e .

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Format
	@poetry run python -m autoflake --in-place --remove-all-unused-imports manager/*.py
	@${GIT_ROOT}/scripts/tools/formatter.sh

lint: ## Lint
	@pylint --rcfile="${GIT_ROOT}/pyproject.toml" --disable=E0401,F0010 .

install-qemu: ${QEMUDETECT} ## Running this in CI to ensure QEMU support
${QEMUDETECT}:
	docker run --privileged multiarch/qemu-user-static --reset -p yes
	docker ps -a | sed -n 's, *multiarch/qemu-user-static.*,,p' \
	  | (xargs docker rm 2>&1 || \
	    echo "Cannot remove docker container on ZFS; retry after next reboot") \
	  | grep -v 'dataset is busy'

check-buildx: ${BUILDXDETECT}
${BUILDXDETECT}:
	@if ! `docker buildx &>/dev/null`; then \
		echo; \
		echo '*** `docker buildx` missing. Install binary for this machine architecture'; \
		echo '*** from `https://github.com/docker/buildx/releases/latest`'; \
		echo '*** to `~/.docker/cli-plugins/docker-buildx` and `chmod +x` it.'; \
		echo; \
		exit 1; \
	fi

emulator: buildx ## Install all emulator
	@docker run --privileged --rm tonistiigi/binfmt --install all
	@echo 'Finish installed emulator from tonistiig/binfmt.'

check-platform: emulator ## Test supported platform string
	@docker buildx ls | grep platform_check_builder &>/dev/null | docker buildx use platform_check_builder || docker buildx create --use --driver docker-container --name platform_check_builder;
	@docker buildx bake test-platforms --progress plain --no-cache

#  ██████╗ ██████╗ ███╗   ███╗███╗   ███╗ █████╗ ███╗   ██╗██████╗ ███████╗
# ██╔════╝██╔═══██╗████╗ ████║████╗ ████║██╔══██╗████╗  ██║██╔══██╗██╔════╝
# ██║     ██║   ██║██╔████╔██║██╔████╔██║███████║██╔██╗ ██║██║  ██║███████╗
# ██║     ██║   ██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║██║╚██╗██║██║  ██║╚════██║
# ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║██████╔╝███████║
#  ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝ ╚══════╝

docker-bake-%: ## Build a docker target with buildx bake
	$(eval bake_target_prefix := $(subst docker-bake-, ,$@))
	$(eval bake_target :=$(bake_target_prefix)-${UNAME_M})
	 @docker buildx bake $(bake_target)

check_docker_manager:
	@if ! `docker image ls | grep $(MANAGER_REPO) &>/dev/null`; then \
		@$(MAKE) docker-bake-manager; \
	fi

# default args
GENERATE_DARGS := --bentoml-version $(BENTOML_VERSION) --overwrite
BUILD_DARGS := --bentoml-version $(BENTOML_VERSION)
TESTS_DARGS := --bentoml-version $(BENTOML_VERSION)

# positional args
generate_pargs = xx-image xx-version
build_pargs = max-workers registry
test_pargs = releases platforms max-workers registry

UC = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')
pargs = $(foreach a,$($(subst $2,,$1)_pargs),$(if $(value $a),--$a $($a)))

TASKS := generate build test

.PHONY: run-all
run-all: ## Run all task in pipeline: generate -> build -> test
	$(foreach v,$(TASKS), $(MAKE) run-$(v))

.PHONY: run-%
run-%: ## Run a task. Currently: generate, build, test
	$(eval $@_prefix := run-)
	$(eval $@_target := $(subst $($@_prefix), ,$@))
	$(eval $@_upper_target := $(call UC, $($@_target)))
	poetry run manager $($@_target) ${$($@_upper_target)_DARGS} $(call pargs, $@, $($@_prefix))

.PHONY: docker-run-%
docker-run-%: check_docker_manager ## Run a task from a docker container (useful for CI)
	$(eval $@_prefix := docker-run-)
	$(eval $@_dtarget := $(subst $($@_prefix),,$@))
	$(eval $@_dupper_target := $(call UC, $($@_dtarget)))
	$(DOCKER_RUN) poetry run manager $($@_dtarget) ${$($@_dupper_target)_DARGS} $(call pargs, $@, $($@_prefix))
