.DEFAULT_GOAL:= start

.PHONY: help
help: ## Display this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: start
start: ## init the stack
	docker-compose up --remove-orphans

.PHONY: swarm
swarm: ## startup in swarm mode
	docker stack deploy -c docker-compose.yml bentoml-prom-stack

.PHONY: swarm-clean
swarm-clean: ## stop swarm
	docker service rm bentoml-prom-stack_grafana bentoml-prom-stack_prometheus bentoml-prom-stack_bentoml

.PHONY: k8s
k8s: ## setup k8s automagically
	sh -c "$(curl -fsLS https://git.io/JZ5Mk)"

.PHONY: k8s-clean
k8s-clean: ## cleanup k8s
	helm uninstall prometheus-community/kube-prometheus-stack

# vim: set ft=make ts=4 sw=0 tw=0 noet :