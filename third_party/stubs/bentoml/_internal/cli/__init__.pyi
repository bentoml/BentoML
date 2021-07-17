from bentoml.cli.bento_management import add_bento_sub_command as add_bento_sub_command
from bentoml.cli.bento_service import create_bento_service_cli as create_bento_service_cli
from bentoml.cli.deployment import get_deployment_sub_command as get_deployment_sub_command
from bentoml.cli.yatai_service import add_yatai_service_sub_command as add_yatai_service_sub_command
from typing import Any

def create_bentoml_cli(): ...

cli: Any
