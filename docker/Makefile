DOCKER_ORG:=aarnphm
FMT?="{{.Repository}}:{{.Tag}}\t{{.Size}}"

.PHONY: help
help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: image-list
image-list: ## Display releases image list
	@docker images --no-trunc -a --filter=reference='model*:*' --format ${FMT} | sort | column -t -s $$'\t'
	@printf "\n\n"
	@docker images --no-trunc -a --filter=reference='yatai*:*' --format ${FMT} | sort | column -t -s $$'\t'
	@printf "\n\n"
	@docker images --no-trunc -a --filter=reference='${DOCKER_ORG}/*:*' --format ${FMT} | sort | column -t -s $$'\t'