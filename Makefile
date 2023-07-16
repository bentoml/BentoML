SHELL := /bin/bash
.DEFAULT_GOAL := help

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'
.PHONY: .pdm
.pdm:  ## Check that PDM is installed
	@pdm -V || echo 'Please install PDM: https://pdm.fming.dev/latest/\#installation'
.PHONY: .pre-commit
.pre-commit:  ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'
.PHONY: install
install: .pdm .pre-commit  ## Install the package, dependencies, and pre-commit for local development
	pdm install -G docs -G tooling -G testing -G io -G grpc -G triton -G tracing -G monitor-otlp -G grpc-reflection -G grpc-channelz -G aws
	pre-commit install --install-hooks
.PHONY: refresh-lockfiles
refresh-lockfiles: .pdm  ## Sync lockfiles with requirements files.
	pdm update --update-reuse -G docs -G tooling -G testing -G io -G grpc -G triton -G tracing -G monitor-otlp -G grpc-reflection -G grpc-channelz -G aws
.PHONY: rebuild-lockfiles
rebuild-lockfiles: .pdm  ## Rebuild lockfiles from scratch, updating all dependencies
	pdm update --update-eager -G docs -G tooling -G testing -G io -G grpc -G triton -G tracing -G monitor-otlp -G grpc-reflection -G grpc-channelz -G aws
.PHONY: format format-proto lint lint-proto type style clean
format: ## Running code formatter: black and isort
	@echo "(black) Formatting codebase..."
	@pre-commit run --all-files black-jupyter
format-proto: ## Running proto formatter: buf
	@echo "Formatting proto files..."
	docker run --init --rm --volume $(GIT_ROOT)/src:/workspace --workdir /workspace bufbuild/buf format --config "/workspace/bentoml/grpc/buf.yaml" -w bentoml/grpc
lint: ## Running lint checker: ruff
	@echo "(ruff) Linting development project..."
	@pre-commit run --all-files ruff
lint-proto: ## Running proto lint checker: buf
	@echo "Linting proto files..."
	docker run --init --rm --volume $(GIT_ROOT)/src:/workspace --workdir /workspace bufbuild/buf lint --config "/workspace/bentoml/grpc/buf.yaml" --error-format msvs bentoml/grpc
type: ## Running type checker: pyright
	@echo "(pyright) Typechecking codebase..."
	@pre-commit run typecheck --all-files
style: format lint format-proto lint-proto ## Running formatter and linter
clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	@cd $(GIT_ROOT)/docs && make clean
	@cd $(GIT_ROOT) || exit 1
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

# Docs
watch-docs: ## Build and watch documentation
	sphinx-autobuild docs/source docs/build/html --watch $(GIT_ROOT)/src/ --ignore "bazel-*"
spellcheck-docs: ## Spell check documentation
	sphinx-build -b spelling ./docs/source ./docs/build || (echo "Error running spellchecker.. You may need to run 'make install-spellchecker-deps'"; exit 1)

OS := $(shell uname)
ifeq ($(OS),Darwin)
install-spellchecker-deps: ## Install MacOS dependencies for spellchecker
	brew install enchant
	pip install sphinxcontrib-spelling
else ifneq ("$(wildcard $(/etc/debian_version))","")
install-spellchecker-deps: ## Install Debian-based dependencies for spellchecker
	sudo apt install libenchant-dev
else
install-spellchecker-deps: ## Inform users to install enchant depending on their distros
	@echo Make sure to install enchant from your distros package manager
	@exit 1
endif
