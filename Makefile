SHELL := /bin/bash
.DEFAULT_GOAL := help

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
USE_VERBOSE ?=false
USE_GPU ?= false
USE_GRPC ?= false

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format format-proto lint lint-proto type style clean
format: ## Running code formatter: black and isort
	@echo "(black) Formatting codebase..."
	@black --config pyproject.toml src tests docs examples
	@echo "(black) Formatting stubs..."
	@find src -name "*.pyi" ! -name "*_pb2*" -exec black --pyi --config pyproject.toml {} \;
	@echo "(isort) Reordering imports..."
	@isort .
	@echo "(ruff) Running fix only..."
	@ruff check src examples tests --fix-only
format-proto: ## Running proto formatter: buf
	@echo "Formatting proto files..."
	docker run --init --rm --volume $(GIT_ROOT)/src:/workspace --workdir /workspace bufbuild/buf format --config "/workspace/bentoml/grpc/buf.yaml" -w bentoml/grpc
lint: ## Running lint checker: ruff
	@echo "(ruff) Linting development project..."
	@ruff check src examples tests
lint-proto: ## Running proto lint checker: buf
	@echo "Linting proto files..."
	docker run --init --rm --volume $(GIT_ROOT)/src:/workspace --workdir /workspace bufbuild/buf lint --config "/workspace/bentoml/grpc/buf.yaml" --error-format msvs bentoml/grpc
type: ## Running type checker: pyright
	@echo "(pyright) Typechecking codebase..."
	@pyright -p src -w
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
