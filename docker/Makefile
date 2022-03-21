UNAME_M ?= $(shell uname -m)
GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
BENTOML_VERSION := $(shell git describe --tags `git rev-list --tags --max-count=1` | sed 's/v\(\)/\1/; s/-//g')

# ----------------------
.DEFAULT_GOAL := help

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install-local
install-local: ## Install manager locally
	@pip install -e .

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
	@find . -name '__pycache__' -type d -exec rm -rf '{}';\

# ----------------------
#
include hack/docker.mk

# default args
GENERATE_DARGS := --bentoml-version $(BENTOML_VERSION) --overwrite
BUILD_DARGS := --bentoml-version $(BENTOML_VERSION) --max-workers $(MAX_WORKERS)

# positional args
generate_pargs = organization
build_pargs = distros python-version releases organization

# functions
pargs = $(foreach a,$($(subst $2,,$1)_pargs),$(if $(value $a),--$a $($a)))
expands = $(foreach a,$2, $1=$a)

.PHONY: run-%
run-%: ## Run a task. Currently: generate, build, test
	$(eval $@_prefix := run-)
	$(eval $@_target := $(subst $($@_prefix), ,$@))
	$(eval $@_upper_target := $(call upper, $($@_target)))
	poetry run manager $($@_target) ${$($@_upper_target)_DARGS} $(call pargs, $@, $($@_prefix))

# .PHONY: docker-run-test
# docker-run-test: ## Run test
# 	docker run $(COMMON_DOCKER_ARGS) -v $(PWD):/work ${ORG}/bats-test:latest ./hack/runt

# .PHONY: docker-run-%
# docker-run-%: ## Run a task from a docker container (useful for CI)
# 	@if ! `docker image ls | grep $(MANAGER_REPO) &>/dev/null` ; then \
# 		docker pull $(FULL_TAG); \
# 	fi
# 	$(eval $@_prefix := docker-run-)
# 	$(eval $@_target := $(subst $($@_prefix), ,$@))
# 	$(eval $@_upper_target := $(call upper, $($@_target)))
# 	$(MANAGER_DOCKER_RUN) poetry run manager $($@_target) ${$($@_upper_target)_DARGS} $(call pargs, $@, $($@_prefix))

.PHONY: all
all:
	$(eval $@_DISTROS := amazonlinux2 alpine3.14 ubi8 debian10 debian11)
	$(eval $@_RELEASES:= all)
	$(eval $@_distro_args := $(call expands,distros,$($@_DISTROS)))
	$(eval $@_releases_args := $(call expands,releases,$($@_RELEASES)))
	$(MAKE) docker-run-generate organization=$(ORG)
	$(MAKE) run-build organization=$(ORG) $($@_distro_args) $($@_releases_args)

%:
	@true
