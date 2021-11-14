.DEFAULT_GOAL := help

GIT_ROOT=$(shell git rev-parse --show-toplevel)

CHECKER_IMG := bentoml/checker:1.0
BASE_ARGS := -i --rm --user root --volume $(GIT_ROOT):/bentoml

CNTR_ARGS := $(BASE_ARGS) $(CHECKER_IMG)
CMD := docker run $(CNTR_ARGS)
TTY := docker run -t $(CNTR_ARGS)

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'

build-checker-img: ## Build checker images
	@if [[ `git diff $(GIT_ROOT)/docker/Dockerfile` != "" ]]; then \
		docker build -f ./scripts/Dockerfile-checker -t $(CHECKER_IMG) . ;\
		docker push $(CHECKER_IMG); \
	fi

pull-checker-img: ## Pull checker images
	@if [[ `docker images --filter=reference='bentoml/checker' -q` == "" ]]; then \
		echo "Pulling bentoml/checker:1.0..."; \
	    docker pull bentoml/checker:1.0; \
	fi \

format: pull-checker-img ## Running code formatter: black and isort
	$(CMD) ./scripts/tools/formatter.sh
lint: pull-checker-img ## Running lint checker: flake8 and pylint
	$(CMD) ./scripts/tools/linter.sh
type: pull-checker-img ## Running type checker: mypy and pyright
	$(CMD) ./scripts/tools/type_checker.sh

__style_src := $(wildcard ./scripts/ci/style/*.sh)
__style_name := ${__style_src:_check.sh=}
__cmd :=$(foreach t, $(__style_name), ci-$(shell basename $(t)))
__filters=ci-docs_spell
tools := $(filter-out $(__filters),$(__cmd))

ci-all: $(tools) ## Running codestyle in CI: black, isort, flake8, pylint, mypy, pyright

ci-%: build-checker-img pull-checker-img
	$(eval style := $(subst ci-, ,$@))
	$(CMD) ./scripts/ci/style/$(style)_check.sh

.PHONY: ci-format
ci-format: ci-black ci-isort ## Running format check in CI: black, isort

.PHONY: ci-lint
ci-lint: ci-flake8 ci-pylint ## Running lint check in CI: flake8, pylint

.PHONY: ci-type
ci-type: ci-mypy ci-pyright ## Running type check in CI: mypy, pyright


install-local: ## Install BentoML from current directory in editable mode
	pip install --editable .
install-dev-deps: ## Install all dev and tests dependencies
	@echo Ensuring dev dependencies...
	@pip install -e ".[dev]"
install-docs-deps:  ## Install documentation dependencies
	@echo Installing docs dependencies...
	@pip install -e ".[doc_builder]"

# Docs
watch-docs: ## Build and watch documentation
	@./docs/watch.sh || (echo "Error building... You may need to run 'make install-watch-deps'"; exit 1)
spellcheck-doc: ## Spell check documentation
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
install-spellchecker-deps: ## Inform users to install enchant depending on their distros
	@echo Make sure to install enchant from your distros package manager
endif

hooks: ## Install pre-defined hooks
	@./scripts/install_hooks.sh
