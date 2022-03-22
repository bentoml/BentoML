DOCKER ?= docker

ORG ?= aarnphm
TAG ?= 1.1.0

# functions
pargs = $(foreach a, $1, $(if $(value $a),--$a $($a)))
upper = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')
word-dash = $(word $2,$(subst -, ,$1))

# Some more functions from shell
GIT_ROOT ?= $(shell git rev-parse --show-toplevel)

# ----------------------
export DOCKER_BUILDKIT=1

COMMON_ARGS := -u root --privileged --rm --init \
			   -v $(realpath /etc/localtime):/etc/localtime:ro \
			   -v /var/run/docker.sock:/var/run/docker.sock
MANAGER_ARGS := ${COMMON_ARGS} \
				-v ${PWD}:/bentoml \
				-w /bentoml ${ORG}/bentoml-docker:${TAG}
HADOLINT_ARGS := ${COMMON_ARGS} \
				-i -v ${PWD}:/workdir \
				-w /workdir ghcr.io/hadolint/hadolint

COMMON_TEST_ARGS := ${COMMON_ARGS} \
			 		-v ${PWD}:/work/manager \
					-w /work
TEST_RUNTIME_ARGS := ${COMMON_TEST_ARGS} ${ORG}/bentoml-docker:test-runtime
TEST_CUDNN_ARGS := ${COMMON_TEST_ARGS} ${ORG}/bentoml-docker:test-cudnn

docker-run-%: ## Run with predefined args and cmd: make docker-run-hadolint -- /bin/hadolint Dockerfile
	$(eval $@_args := $(call upper, $(call word-dash, $@, 3)))
	$(DOCKER) run ${$($@_args)_ARGS} $(filter-out $@,$(MAKECMDGOALS))

docker-runi-%: ## Run a container interactively
	$(eval $@_args := $(call upper, $(call word-dash, $@, 3)))
	$(DOCKER) run -it ${$($@_args)_ARGS} bash

docker-bake-%: ## Build a docker target with buildx bake: make docker-bake-test
	$(eval $@_target := $(call word-dash, $@, 3))
	$(MAKE) buildx-misc-create -- manager-docker
	$(DOCKER) buildx bake -f ${GIT_ROOT}/docker/docker-bake.hcl $(filter-out $@,$(MAKECMDGOALS)) $($@_target) 

BUILDXDETECT = ${HOME}/.docker/cli-plugins/docker-buildx
emulator: ${BUILDXDETECT} ## Setup emulator for supporting multiarchitecture.
${BUILDXDETECT}:
	$(MAKE) buildx-misc-preflight
	$(MAKE) buildx-misc-qemu

buildx-misc-%: ## Run support tasks for buildx
	$(eval $@_target := $(call word-dash, $@, 3))
	./hack/shells/buildx_misc -$($@_target) $(filter-out $@,$(MAKECMDGOALS))

lint-dockerfile: ## Lint docker with hadolint
	export LINT_ERRORS=0; \
	IFS=$$'\n'; for dockerfile in $(shell git ls-files | { grep Dockerfile || echo ""; }); do \
		$(MAKE) docker-run-hadolint -- /bin/hadolint "$${dockerfile}" || LINT_ERRORS=$$((LINT_ERRORS+1)); \
	done; \
	exit "$${LINT_ERRORS}"

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | sed 's/^[^:]*://g' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-30s\033[0m %s\n", $$1, $$2}'
