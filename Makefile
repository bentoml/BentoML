.DEFAULT_GOAL := help

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# General Development
test: ## Run all unit tests with current Python version and env
	@./ci/unit_tests.sh || (echo "Error running tests... You may need to run 'make install-test-deps'"; exit 1)
format: ## Format code to adhere to BentoML style
	./dev/format.sh
lint: format ## Lint code
	./dev/lint.sh
install-local: ## Install BentoML from current directory in editable mode
	pip install --editable .
	cd yatai && make install-web-deps
	bentoml --version
install-test-deps: ## Install all test dependencies
	@echo Ensuring test dependencies...
	@pip install -e ".[test]"

# Protos
gen-protos: ## Build protobuf for Python and Node
	@./dev/gen-protos-docker.sh

# Docs
watch-docs: ## Build and watch documentation
	@./docs/watch.sh || (echo "Error building... You may need to run 'make install-watch-deps'"; exit 1)
OS := $(shell uname)
install-docs-deps:  ## Install documentation dependencies
	@echo Installing docs dependencies...
	@pip install -e ".[doc_builder]"
ifeq ($(OS),Darwin)
install-watch-deps: ## Install MacOS dependencies for watching docs
	brew install fswatch
else
install-watch-deps: ## Install Debian-based OS dependencies for watching docs
	sudo apt install inotify-tools
endif
OS := $(shell uname)
ifeq ($(OS),Darwin)
install-spellchecker-deps: ## Install MacOS dependencies for spellchecker
	brew install enchant
	pip install sphinxcontrib-spelling
else
install-spellchecker-deps: ## Install Debian-based dependencies for spellchecker
	sudo apt install libenchant-dev
endif
spellcheck-doc: ## Spell check documentation
	sphinx-build -b spelling ./docs/source ./docs/build || (echo "Error running spellchecker.. You may need to run 'make install-spellchecker-deps'"; exit 1)