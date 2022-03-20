UNAME_M ?= $(shell uname -m)
GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
BENTOML_VERSION := $(shell git describe --tags `git rev-list --tags --max-count=1` | sed 's/v\(\)/\1/; s/-//g')

MANAGER_REPO := aarnphm/bentoml-docker
MANAGER_TAG := 1.1.0
MAX_WORKERS := 6

# ----------------------
export DOCKER_BUILDKIT=1

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
	@poetry run python -m autoflake --recursive --in-place --remove-all-unused-imports manager
	@${GIT_ROOT}/scripts/tools/formatter.sh

lint: ## Lint
	@pylint --rcfile="${GIT_ROOT}/pyproject.toml" --disable=E0401,F0010 .

dead: ## ded
	@dead --files manager/

quality: ## Quality check
	@vulture .

clean: ## cleanup
	@./hack/shells/buildx_misc -r manager_docker;
	@find . -name '__pycache__' -type d -exec rm -rf '{}';\

# ----------------------
install-qemu:
	@./hack/shells/buildx_misc --install-qemu

BUILDXDETECT = ${HOME}/.docker/cli-plugins/docker-buildx
check-buildx: ${BUILDXDETECT}
${BUILDXDETECT}:
	@./hack/shells/buildx_misc --preflight

emulator: check-buildx install-qemu ## Install all emulator

.PHONY: docker-bake
docker-bake: ## Building devcontainer
	@./hack/shells/buildx_misc -create manager_docker
	docker buildx bake --push --no-cache

docker-bake-%: ## Build a docker target with buildx bake
	@./hack/shells/buildx_misc -create manager_docker
	$(eval bake_target_prefix := $(subst docker-bake-, ,$@))
	$(eval bake_target :=$(bake_target_prefix))
	@docker buildx bake $(bake_target) --push

# ----------------------
# default args
GENERATE_DARGS := --bentoml-version $(BENTOML_VERSION) --overwrite
BUILD_DARGS := --bentoml-version $(BENTOML_VERSION) --max-workers $(MAX_WORKERS)

# positional args
generate_pargs = organization
build_pargs = distros python-version releases organization

# functions
UC = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')
rm = $(subst $2,,$1)
pargs = $(foreach a,$($(subst $2,,$1)_pargs),$(if $(value $a),--$a $($a)))
expands = $(foreach a,$2, $1=$a)

.PHONY: run-%
run-%: ## Run a task. Currently: generate, build, test
	$(eval $@_prefix := run-)
	$(eval $@_target := $(subst $($@_prefix), ,$@))
	$(eval $@_upper_target := $(call UC, $($@_target)))
	poetry run manager $($@_target) ${$($@_upper_target)_DARGS} $(call pargs, $@, $($@_prefix))

.PHONY: run-test
run-test: ## Run test
	docker run $(COMMON_DOCKER_ARGS) -v $(PWD):/work aarnphm/bats-test:latest ./hack/runt

.PHONY: docker-run-%
docker-run-%: ## Run a task from a docker container (useful for CI)
	@if ! `docker image ls | grep $(MANAGER_REPO) &>/dev/null` ; then \
		docker pull $(MANAGER_REPO); \
	fi
	$(eval $@_prefix := docker-run-)
	$(eval $@_target := $(subst $($@_prefix), ,$@))
	$(eval $@_upper_target := $(call UC, $($@_target)))
	$(MANAGER_DOCKER_RUN) poetry run manager $($@_target) ${$($@_upper_target)_DARGS} $(call pargs, $@, $($@_prefix))

.PHONY: all
all:
	$(eval $@_ORG := aarnphm)
	$(eval $@_DISTROS := amazonlinux2 alpine3.14 ubi8 debian11 debian10)
	$(eval $@_RELEASES:= all)
	$(eval $@_distro_args := $(call expands,distros,$($@_DISTROS)))
	$(eval $@_releases_args := $(call expands,releases,$($@_RELEASES)))
	$(MAKE) docker-run-generate organization=$($@_ORG)
	$(MAKE) run-build organization=$($@_ORG) $($@_distro_args) $($@_releases_args)

%:
	@true
