from bentoml.exceptions import InvalidArgument as InvalidArgument
from typing import Any

logger: Any
SPEC_FIELDS_AVAILABLE_FOR_UPDATE: Any
SAGEMAKER_FIELDS_AVAILABLE_FOR_UPDATE: Any

def deployment_dict_to_pb(deployment_dict): ...
def deployment_yaml_string_to_pb(deployment_yaml_string): ...
