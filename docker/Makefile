DOCKER_ORG ?= $(shell whoami)
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: help
help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install manager Docker image for development
ifneq (, $(wildcard ./.env))
	@touch ./.env
endif
	@printf "Make sure to add Docker credentials according to manifest.yml under ${ROOT_DIR}/.env\n"
	if [[ `docker images --filter=reference='bentoml-docker' -q` == "" ]] | [[ `git diff Dockerfile` != "" ]]; then \
		DOCKER_BUILDKIT=1 docker build -t bentoml-docker -f Dockerfile .; \
	fi;
	@printf "refers to README.md Developing section.\n"

.PHONY: run
run: ## Run manager Docker image
	@docker run --user $(shell id -u):$(shell id -g) -it -v $(shell pwd):/bentoml bentoml-docker bash