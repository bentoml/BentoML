.DEFAULT_GOAL := help

help: ## Show all Makefile targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

# YataiService gRPC
start-yatai-debug: ## Start YataiService in debug mode
	bentoml yatai-service-start --debug || (echo "Error starting... You may need to run 'make install-yatai-deps'"; exit 1)
start-grpcui: ## Start gPRC Web UI
	grpcui -plaintext localhost:50051 || (echo "Error starting... You may need to run 'make install-yatai-deps'"; exit 1)
install-yatai-deps: ## Install dependencies to debug YataiService
	pip install -e ".[dev]"
	go get github.com/fullstorydev/grpcui
	go install github.com/fullstorydev/grpcui/cmd/grpcui@latest

# BentoML Web UI
watch-yatai-web-ui: ## Start BentoML Web UI server in dev mode
	bentoml yatai-service-start --no-ui & \
	cd web_server && yarn dev 127.0.0.1:50051 3000 . 127.0.0.1:50052
build-yatai-web-ui: ## Build BentoML Web UI server and frontend
	cd web_server && yarn build
run-yatai-web-ui: ## Run production BentoML Web UI server and frontend
	bentoml yatai-service-start --no-ui & \
	cd web_server && yarn start 127.0.0.1:50051 3000 . 127.0.0.1:50052
install-web-deps: ## Install dependencies to run web server and frontend
	cd web_server && yarn install
	cd ui && yarn install