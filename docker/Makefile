UNAME_M ?= $(shell uname -m)
GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
BENTOML_VERSION := $(shell git describe --tags `git rev-list --tags --max-count=1` | sed 's/v\(\)/\1/; s/-//g')

MANAGER_REPO := aarnphm/bentoml-docker
MANAGER_TAG := 1.1.0

# ----------------------
export DOCKER_BUILDKIT=1

BUILDXDETECT = ${HOME}/.docker/cli-plugins/docker-buildx
# Just one of the many files created
QEMUDETECT = /proc/sys/fs/binfmt_misc/qemu-aarch64

FULL_TAG := ${MANAGER_REPO}:${MANAGER_TAG}

COMMON_DOCKER_ARGS := -u root --privileged --rm --init \
	-v $(realpath /etc/localtime):/etc/localtime:ro \
	-v /var/run/docker.sock:/var/run/docker.sock

MANAGER_DOCKER_ARGS := ${COMMON_DOCKER_ARGS} \
	-v ${PWD}:/bentoml \
	-w /bentoml ${FULL_TAG}
MANAGER_DOCKER_RUN := docker run ${MANAGER_DOCKER_ARGS}


# ----------------------
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
	@if [[ "${UNAME_M}" != 'arm64' ]]; then \
		docker run --rm --privileged multiarch/qemu-user-static --reset -p yes; \
	else \
		docker run --rm --privileged aptman/qus -s -- -p i386 arm ppc64le mips64 mips64el s390x aarch64 riscv64 x86_64; \
	fi
remove-qemu: ## Unset registered format
	docker run --rm --privileged aptman/qus -- -r
	docker run --privileged --rm tonistiigi/binfmt --uninstall qemu-*

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

emulator: check-buildx install-qemu ## Install all emulator
	@docker run --privileged --rm tonistiigi/binfmt --install all
	@echo 'Finish installed emulator from tonistiig/binfmt.'

check-platform: emulator ## Test supported platform string
	@docker buildx ls | grep platform_check_builder &>/dev/null | docker buildx use platform_check_builder || docker buildx create --use --driver docker-container --name platform_check_builder;
	@docker buildx bake test-platforms --progress plain --no-cache


# ----------------------
docker-bake-%: ## Build a docker target with buildx bake
	$(eval bake_target_prefix := $(subst docker-bake-, ,$@))
	$(eval bake_target :=$(bake_target_prefix))
	 @docker buildx bake $(bake_target) --push

check_docker_manager:
	@if ! `docker image ls | grep $(MANAGER_REPO) &>/dev/null` || git diff ./hack/dockerfiles/dev.Dockerfile; then \
		$(MAKE) docker-bake-all; \
	fi

# default args
GENERATE_DARGS := --bentoml-version $(BENTOML_VERSION) --overwrite
BUILD_DARGS := --bentoml-version $(BENTOML_VERSION)
TESTS_DARGS := --bentoml-version $(BENTOML_VERSION)
AUTHENTICATE_DARGS := dockerhub

# positional args
generate_pargs = organization
build_pargs = max-workers registry distros python-version releases organization
test_pargs = releases platforms max-workers registry

# flags only
FLAGS = verbose quiet dry-run

UC = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')
rm = $(subst $2,,$1)
pargs = $(foreach a,$($(subst $2,,$1)_pargs),$(if $(value $a),--$a $($a)))
flags_only = $(foreach l,$(FLAGS),$(if $(findstring $l, $1),--$l))

TASKS := generate build

.PHONY: run-all
run-all: ## Run all task in pipeline: generate -> build -> test
	$(foreach v,$(TASKS), $(MAKE) run-$(v))

.PHONY: docker-run-all
docker-run-all: ## Run all task in pipeline: generate -> build -> test
	$(foreach v,$(TASKS), $(MAKE) docker-run-$(v))

.PHONY: run-%
run-%: ## Run a task. Currently: generate, build, test
	$(eval $@_prefix := run-)
	$(eval $@_target := $(subst $($@_prefix), ,$@))
	$(eval $@_upper_target := $(call UC, $($@_target)))
	@poetry run manager $($@_target) ${$($@_upper_target)_DARGS} $(call pargs, $@, $($@_prefix)) $(call flags_only, $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS)))

.PHONY: run-test
run-test: ## Run test
	@docker run $(COMMON_DOCKER_ARGS) -v $(PWD):/work aarnphm/bats-test:latest ./hack/runt

.PHONY: docker-run-%
docker-run-%: check_docker_manager ## Run a task from a docker container (useful for CI)
	$(eval $@_prefix := docker-run-)
	$(eval $@_dtarget := $(subst $($@_prefix), ,$@))
	$(eval $@_dupper_target := $(call UC, $($@_dtarget)))
	@$(MANAGER_DOCKER_RUN) poetry run manager $($@_dtarget) ${$($@_dupper_target)_DARGS} $(call pargs, $@, $($@_prefix)) $(call flags_only, $(wordlist 2, $(words $(MAKECMDGOALS)), $(MAKECMDGOALS)))
%:
	@true
