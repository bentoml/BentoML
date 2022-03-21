SHELL ?=/bin/bash
DOCKER ?= docker

ORG ?= aarnphm
TAG ?= 1.1.0
MAX_WORKERS ?= 6

upper = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')
rm = $(subst $1, ,$2)

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

docker-run-%:
	$(eval $@_args := $(call upper, $(call rm, docker-run-, $@)))
	$(DOCKER) run ${$($@_args)_ARGS} $(filter-out $@,$(MAKECMDGOALS))

# ----------------------
docker-lint: ## Lint docker with hadolint
	export LINT_ERRORS=0; \
	IFS=$$'\n'; for dockerfile in $(shell git ls-files | { grep Dockerfile || echo ""; }); do \
		docker run --rm -i -v "$$(pwd)":/workdir -w /workdir ghcr.io/hadolint/hadolint /bin/hadolint "$${dockerfile}" || LINT_ERRORS=$$((LINT_ERRORS+1)); \
	done; \
	exit "$${LINT_ERRORS}"

docker-buildx-clean: ## clean docker
	@./hack/shells/buildx_misc -r manager_docker

# ----------------------
install-qemu:
	@./hack/shells/buildx_misc -install-qemu

BUILDXDETECT = ${HOME}/.docker/cli-plugins/docker-buildx
check-buildx: ${BUILDXDETECT}
${BUILDXDETECT}:
	@./hack/shells/buildx_misc -preflight

emulator: check-buildx install-qemu ## Install all emulator

docker-bake-%: ## Build a docker target with buildx bake
	@./hack/shells/buildx_misc -create manager_docker
	$(eval bake_target_prefix := $(subst docker-bake-, ,$@))
	$(eval bake_target :=$(bake_target_prefix))
	@docker buildx bake $(bake_target) --push --no-cache
