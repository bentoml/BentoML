SHELL := /bin/bash
.DEFAULT_GOAL := help

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format lint type clean test
format: ## Running code formatter: black and isort
	@./tools/style
lint: ## Running lint checker: pylint
	@./tools/lint
type: ## Running type checker: pyright
	@echo "(pyright) Watching codebase's type..."
	@bazel run //:pyright -- -p $(GIT_ROOT)/src/ -w
test: ## Running tests
	@bazel test //tests/...
clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	@cd $(GIT_ROOT)/docs && make clean
	@find $(GIT_ROOT) -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@bazel clean --expunge

# Docs
watch-docs: ## Build and watch documentation
	@bazel run //:sphinx-autobuild -- $(GIT_ROOT)/docs/source $(GIT_ROOT)/docs/build/html --watch $(GIT_ROOT)/src/ --ignore "$(GIT_ROOT)/bazel-*"
spellcheck-docs: ## Spell check documentation
	@bazel run //:sphinx-build -- -b spelling $(GIT_ROOT)/docs/source $(GIT_ROOT)/docs/build || (echo "Error running spellchecker.. You may need to run 'make install-spellchecker-deps'"; exit 1)

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
