SHELL := /bin/bash
.DEFAULT_GOAL := help

GIT_ROOT ?= $(shell git rev-parse --show-toplevel)
USE_VERBOSE ?=false
AS_ROOT ?= false

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

format: ## Running code formatter: black and isort
	./scripts/tools/formatter.sh
lint: ## Running lint checker: flake8 and pylint
	./scripts/tools/linter.sh
type: ## Running type checker: pyright
	./scripts/tools/type_checker.sh


__style_src := $(wildcard $(GIT_ROOT)/scripts/ci/style/*.sh)
__style_name := ${__style_src:_check.sh=}
tools := $(foreach t, $(__style_name), ci-$(shell basename $(t)))

ci-all: $(tools) ## Running codestyle in CI: black, isort, flake8, pylint, pyright

ci-%:
	$(eval style := $(subst ci-, ,$@))
	@./scripts/ci/style/$(style)_check.sh

.PHONY: ci-format
ci-format: ci-black ci-isort ## Running format check in CI: black, isort

.PHONY: ci-lint
ci-lint: ci-flake8 ci-pylint ## Running lint check in CI: flake8, pylint


tests-%:
	$(eval type :=$(subst tests-, , $@))
	$(eval RUN_ARGS:=$(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)))
	$(eval __positional:=$(foreach t, $(RUN_ARGS), --$(t)))
ifeq ($(USE_VERBOSE),true)
	./scripts/ci/run_tests.sh -v $(type) $(__positional)
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
	@./scripts/watch_docs.sh || (echo "Error building... You may need to run 'make install-watch-deps'"; exit 1)
spellcheck-docs: ## Spell check documentation
	sphinx-build -b spelling ./docs/source ./docs/build || (echo "Error running spellchecker.. You may need to run 'make install-spellchecker-deps'"; exit 1)

OS := $(shell uname)
ifeq ($(OS),Darwin)
install-watch-deps: ## Install MacOS dependencies for watching docs
	brew install fswatch
install-spellchecker-deps: ## Install MacOS dependencies for spellchecker
	brew install enchant
	pip install sphinxcontrib-spelling
else ifneq ("$(wildcard $(/etc/debian_version))","")
install-watch-deps: ## Install Debian-based OS dependencies for watching docs
	sudo apt install inotify-tools
install-spellchecker-deps: ## Install Debian-based dependencies for spellchecker
	sudo apt install libenchant-dev
else
install-watch-deps: ## Inform users to install inotify-tools depending on their distros
	@echo Make sure to install inotify-tools from your distros package manager
	@exit 1
install-spellchecker-deps: ## Inform users to install enchant depending on their distros
	@echo Make sure to install enchant from your distros package manager
	@exit 1
endif

hooks: ## Install pre-defined hooks
	@./scripts/install_hooks.sh

check_defined = $(strip $(foreach 1,$1, $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
        $(error Undefined $1$(if $2, ($2))$(if $(value @), \
                required by target `$@`)))
check-defined-% : __check_defined_FORCE
	$(eval $@_target := $(subst check-defined-, ,$@))
	@:$(call check_defined, $*, $@_target)

.PHONY : __check_defined_FORCE
__check_defined_FORCE:
