SHELL := /bin/bash
.DEFAULT_GOAL := help

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
USE_VERBOSE ?=false
USE_GPU ?= false

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: format lint type style clean
format: ## Running code formatter: black and isort
	@./scripts/tools/formatter.sh
lint: ## Running lint checker: pylint
	@./scripts/tools/linter.sh
type: ## Running type checker: pyright
	@./scripts/tools/type_checker.sh
style: format lint ## Running formatter and linter
clean: ## Clean all generated files
	@echo "Cleaning all generated files..."
	@cd $(GIT_ROOT)/docs && make clean
	@cd $(GIT_ROOT) || exit 1
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete


ci-%:
	$(eval style := $(subst ci-, ,$@))
	@./scripts/ci/style/$(style)_check.sh

.PHONY: ci-format
ci-format: ci-black ci-isort ## Running format check in CI: black, isort

.PHONY: ci-lint
ci-lint: ci-pylint ## Running lint check in CI: pylint

.PHONY: tests-suite
tests-suite: tests-unit tests-general_features ## Running BentoML tests suite (unit, e2e, integration)

tests-%:
	$(eval type :=$(subst tests-, , $@))
	$(eval RUN_ARGS:=$(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)))
	$(eval __positional:=$(foreach t, $(RUN_ARGS), -$(t)))
ifeq ($(USE_VERBOSE),true)
	./scripts/ci/run_tests.sh -v $(type) $(__positional)
else ifeq ($(USE_GPU),true)
	./scripts/ci/run_tests.sh -v $(type) --gpus $(__positional)
else
	./scripts/ci/run_tests.sh $(type) $(__positional)
endif


install-local: ## Install BentoML in editable mode
	@pip install --editable .
install-dev-deps: ## Install all dev dependencies
	@echo Installing dev dependencies...
	@pip install -r requirements/dev-requirements.txt
install-tests-deps: ## Install all tests dependencies
	@echo Installing tests dependencies...
	@pip install -r requirements/tests-requirements.txt
install-docs-deps: ## Install documentation dependencies
	@echo Installing docs dependencies...
	@pip install -r requirements/docs-requirements.txt

# Docs
watch-docs: install-docs-deps ## Build and watch documentation
	sphinx-autobuild docs/source docs/build/html --watch $(GIT_ROOT)/bentoml
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
