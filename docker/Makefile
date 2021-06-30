SHELL=/bin/bash

FORMAT ?= "{{.Repository}}:{{.Tag}}\t{{.Size}}"
DOCKER_ORG ?= $(shell whoami)
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


.PHONY: help
help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: list
list: ## Display list of built images
	@echo "model-server"
	@echo "==============================================================="
	@docker images --no-trunc -a --filter=reference='${DOCKER_ORG}/model*:*' --format ${FORMAT} | sort | column -t -s $$'\t'
	@printf "\n\n"
	@echo "yatai-service"
	@echo "==============================================================="
	@docker images --no-trunc -a --filter=reference='${DOCKER_ORG}/yatai*:*' --format ${FORMAT} | sort | column -t -s $$'\t'

.PHONY: install
install: ## Install manager Docker image for development
ifneq (, $(wildcard ./.env))
	@touch ./.env
endif
	$(warning Make sure to add Docker credentials according to manifest.yml under ${ROOT_DIR}/.env)
	@echo
	if [[ `docker images --filter=reference='bentoml-docker' -q` == "" ]] | [[ `git diff Dockerfile` != "" ]]; then \
		docker build -t bentoml-docker -f Dockerfile .; \
	fi;
	source ./tools/dev.sh

.PHONY: run
run: ## Run manager Docker image
	@docker run --user $(shell id -u):$(shell id -g) -i -t -v $(shell pwd):/bentoml bentoml-docker bash