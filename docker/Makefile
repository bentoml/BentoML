# ----------------------
include hack/docker.mk

# ----------------------
.DEFAULT_GOAL := help

# Set the shell
SHELL := /bin/bash

USE_DOCKER ?= true
# type of releases: base, runtime, cudnn, or `all` of them :/
RELEASES ?= all
# Default distros that we releases
DISTROS ?= amazonlinux2 alpine3.14 ubi8 debian10 debian11
# number of max workers to use
MAX_WORKERS ?= 6

# test python version
TEST_PYTHON_VERSION ?= 3.10

# ----------------------
BENTOML_VERSION := $(shell git describe --tags `git rev-list --tags --max-count=1` | sed 's/v\(\)/\1/; s/-//g')

.PHONY: install
install: ## Install manager locally
	@pip install -e .

format: ## Format
	@poetry run python -m autoflake --recursive --in-place --remove-all-unused-imports manager
	@${GIT_ROOT}/scripts/tools/formatter.sh

lint: ## Lint
	@pylint --rcfile="${GIT_ROOT}/pyproject.toml" --disable=E0401,F0010 .
ifeq ($(USE_DOCKER),true)
	$(MAKE) lint-dockerfile
endif

dead: ## deadcode
	@dead --files manager/

quality: ## Quality check
	@vulture .

# ----------------------
# Run some manager commands

POETRY_RUN := poetry run manager $(1)

# default args
GENERATE_DEFAULTS := --bentoml-version $(BENTOML_VERSION) --overwrite
BUILD_DEFAULTS := --bentoml-version $(BENTOML_VERSION) --max-workers $(MAX_WORKERS)

# positional args
GENERATE_POSITIONAL := organization
BUILD_POSITIONAL := distros python-version releases organization

.PHONY: run-%
run-%: ## Run a task. Currently: generate, build, test
	$(eval $@_target := $(call word-dash, $@, 2))
	$(eval $@_upper := $(call upper, $($@_target)))
	$(eval $@_ARGS := $($@_target) ${$($@_upper)_DEFAULTS} $(call pargs, ${$($@_upper)_POSITIONAL}))
ifeq ($(USE_DOCKER),true)
	$(MAKE) docker-run-manager -- $(POETRY_RUN) $($@_ARGS)
else
	$(POETRY_RUN) $($@_ARGS)
endif

.PHONY: run-test
run-test: ## Test
	$(MAKE) buildx-misc-genvar $(TEST_PYTHON_VERSION)
	$(MAKE) docker-bake-test -- -f ./env.hcl

.PHONY: all
all: ## To release all or not release all?
	$(MAKE) run-generate organization=$(ORG)
	$(MAKE) run-build organization=$(ORG) $(call expands,distros,$(DISTROS)) $(call expands,releases,$(RELEASES))

%:
	@true
