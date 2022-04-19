# ----------------------
include hack/docker.mk

# ----------------------
.DEFAULT_GOAL := help

# Set the shell
SHELL := /bin/bash

# TODO: authentication issue with buildkit when using docker to push
USE_DOCKER ?= false
# type of releases: base, runtime, cudnn, or `all` of them :/
RELEASES ?= all
# Default distros that we releases
DISTROS ?= amazonlinux2
# Default python version
PYTHON_VERSION ?= 3.7 3.8 3.9 3.10
# number of max workers to use
MAX_WORKERS ?= 6

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
pargs = $(foreach a, $1, $(if $(value $a),--$a $($a)))

POETRY_RUN := poetry run manager $(1)

# default args
GENERATE_DEFAULTS := --bentoml-version $(BENTOML_VERSION)
BUILD_DEFAULTS := --bentoml-version $(BENTOML_VERSION) --max-workers $(MAX_WORKERS)

# positional args
GENERATE_POSITIONAL := organization
BUILD_POSITIONAL := releases organization

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
ifeq ($(TEST_TYPE),runtime)
	docker pull ${ORG}/bentoml-docker:test-runtime
else
	docker pull ${ORG}/bentoml-docker:test-cudnn
endif
	export TEST_ERRORS=0; \
	IFS=$$'\n'; for python_version in ${PYTHON_VERSION}; do \
		for distros in ${DISTROS}; do \
			$(MAKE) docker-run-test -- -default -platforms $(shell uname -m) -python_version "$${python_version}" -distros "$${distros}" -suffix ${TEST_TYPE} -organization ${ORG} -bentoml_version ${BENTOML_VERSION} || TEST_ERRORS=$$((TEST_ERRORS+1)); \
		done; \
	done; \
	exit "$${TEST_ERRORS}"


.PHONY: all
all: ## To release all or not release all?
	@set -x;
	@$(MAKE) run-generate -- organization=$(ORG);
	@IFS=$$'\n'; for DISTRO in ${DISTROS}; do \
		for RELEASE in ${RELEASES}; do \
			$(MAKE) run-build -- organization=$(ORG) distros="$${DISTRO}" releases="$${RELEASE}"; \
		done; \
	done; \

%:
	@true
