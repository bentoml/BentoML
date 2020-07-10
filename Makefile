.DEFAULT_GOAL := help

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# General Development
test: ## Run all unit tests with current Python version and env
	@./travis/unit_tests.sh || (echo "Error running tests... You may need to run 'make install-test-deps'"; exit 1)
format: ## Format code to adhere to BentoML style
	./dev/format.sh
lint: ## Lint code
	./dev/lint.sh
install-local: ## Install BentoML from current directory in editable mode
	pip install --editable .
	bentoml --version
install-test-deps: ## Install all test dependencies
	@echo Ensuring test dependencies...
	@pip install -e ".[test]" --quiet

# Docs
watch: ## Build and watch documentation
	@./docs/watch.sh || (echo "Error building... You may need to run 'make install-watch-deps'"; exit 1)
OS := $(shell uname)
ifeq ($(OS),Darwin)
install-watch-deps: ## Install MacOS dependencies for watching docs
	brew install fswatch
else
install-watch-deps: ## Install Linux dependencies for watching docs
	sudo apt install inotify-tools
endif

# YataiService gRPC
yatai: ## Start YataiService in debug mode
	bentoml yatai-service-start --debug || (echo "Error starting... You may need to run 'make install-yatai-deps'"; exit 1)
grpcui: ## Start gPRC Web UI
	grpcui -plain text localhost:50051 || (echo "Error starting... You may need to run 'make install-yatai-deps'"; exit 1)
install-yatai-deps: ## Install dependencies to debug YataiService
	pip install -e ".[dev]"
	go get github.com/fullstorydev/grpcui
	go install github.com/fullstorydev/grpcui/cmd/grpcui

# BentoML Web UI
web-ui: ## Build BentoML Web UI server and frontend
	cd bentoml/yatai/web && npm run build
install-web-deps: ## Install dependencies to run web server and frontend
	cd bentoml/yatai/web && yarn install
	cd bentoml/yatai/web/client && yarn install