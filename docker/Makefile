DOCKER_REGISTRY:= docker.io
DOCKER_ORG:= bentoml
DOCKER_IMAGE:= model-server
DOCKER_FULL:=$(DOCKER_REGISTRY)/$(DOCKER_ORG)/$(DOCKER_IMAGE)

.PHONY: analyze
analyze:
	@dive ${IMAGE_TAG}