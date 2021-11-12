.DEFAULT_GOAL := help

CHECKER_IMG := bentoml/checker:1.0
BASE_ARGS := -u root -v $(PWD):/bentoml

CNTR_ARGS := $(BASE_ARGS) $(CHECKER_IMG)
CMD := docker run $(CNTR_ARGS)
TTY := docker run -it $(CNTR_ARGS)

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

build-checker-img: ## Build checker images
	@docker build -f ./scripts/Dockerfile-checker -t $(CHECKER_IMG) . || exit 1
	@docker push $(CHECKER_IMG)

pull-checker-img: ## Pull checker images
	@docker pull $(CHECKER_IMG) || true

format: pull-checker-img ## Format code
	$(CMD) ./scripts/tools/formatter.sh
lint: pull-checker-img ## Lint code
	$(CMD) ./scripts/tools/linter.sh
type: pull-checker-img ## Running type checker, mypy and pyright
	$(CMD) ./scripts/tools/type_checker.sh

ci-black: pull-checker-img ## Running black in CI
	$(CMD) ./scripts/ci/style/black_check.sh
ci-isort: pull-checker-img ## Running isort in CI
	$(CMD) ./scripts/ci/style/isort_check.sh
ci-format: ci-black ci-isort ## Running format check in CI

ci-flake8: pull-checker-img ## Running flake8 in CI
	$(CMD) ./scripts/ci/style/flake8_check.sh
ci-pylint: pull-checker-img ## Running pylint in CI
	$(CMD) ./scripts/ci/style/pylint_check.sh
ci-lint: ci-flake8 ci-pylint ## Running lint check in CI

ci-mypy: pull-checker-img ## Running mypy in CI
	$(CMD) ./scripts/ci/style/mypy_check.sh
ci-pyright: pull-checker-img ## Running pyright in CI
	$(CMD) ./scripts/ci/style/pyright_check.sh
ci-type: ci-mypy ci-pyright ## Running type check in CI


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