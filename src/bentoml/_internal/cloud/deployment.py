from .config import get_rest_api_client
from .config import default_context_name
from .config import default_kube_namespace
from .schemas import DeploymentSchema
from .schemas import schema_from_json
from .schemas import DeploymentListSchema
from .schemas import CreateDeploymentSchema
from ...exceptions import BentoMLException


class deployment:
    def list_deployment(
        self, context: str | None = None, clusterName: str | None = None
    ) -> DeploymentListSchema:

        yatai_rest_client = get_rest_api_client(context)
        if clusterName:
            res = yatai_rest_client.get_deployment_list(clusterName)
        else:
            res = yatai_rest_client.get_deployment_list(default_context_name)
        if res is None:
            raise BentoMLException("List deployments request failed")
        return res

    def create_deployment(
        self,
        context: str | None = None,
        clusterName: str | None = None,
        *,
        json_content: str,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if clusterName is None:
            clusterName = default_context_name
        deployment_schema = schema_from_json(json_content, CreateDeploymentSchema)
        for target in deployment_schema.targets:
            res = yatai_rest_client.get_bento(target.bento_repository, target.bento)
            if res is None:
                raise BentoMLException(
                    f"Create deployment: {target.bento_repository}:{target.bento} does not exist"
                )
        res = yatai_rest_client.get_deployment(
            clusterName, deployment_schema.kube_namespace, deployment_schema.name
        )
        if res is not None:
            raise BentoMLException("Create deployment: Deployment already exists")
        res = yatai_rest_client.create_deployment(clusterName, json_content)
        if res is None:
            raise BentoMLException("Create deployment request failed")
        return res

    def get_deployment(
        self,
        context: str | None = None,
        clusterName: str | None = None,
        kubeNamespace: str | None = None,
        *,
        deploymentName: str,
    ) -> DeploymentSchema:

        yatai_rest_client = get_rest_api_client(context)
        if clusterName is None:
            clusterName = default_context_name
        if kubeNamespace is None:
            kubeNamespace = default_kube_namespace
        res = yatai_rest_client.get_deployment(
            clusterName, kubeNamespace, deploymentName
        )
        if res is None:
            raise BentoMLException("Get deployment request failed")
        return res
