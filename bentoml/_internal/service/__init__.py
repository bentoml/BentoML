import logging
import re
from typing import List

# from ..adapters import BaseInputAdapter, BaseOutputAdapter, DefaultOutput
from ..exceptions import InvalidArgument, NotFound

# from bentoml.saved_bundle.config import (
#     DEFAULT_MAX_BATCH_SIZE,
#     DEFAULT_MAX_LATENCY,
#     SavedBundleConfig,
# )
# from bentoml.saved_bundle.pip_pkg import seek_pip_packages
# from bentoml.service.artifacts import ArtifactCollection, ServiceArtifact
# from bentoml.service.env import ServiceEnv
from ..service.inference_api import InferenceAPI

BENTOML_RESERVED_API_NAMES = [
    "index",
    "swagger",
    "docs",
    "healthz",
    "metrics",
    "feedback",
]
logger = logging.getLogger(__name__)
prediction_logger = logging.getLogger("bentoml.prediction")


def validate_inference_api_name(api_name: str):
    if not api_name.isidentifier():
        raise InvalidArgument(
            "Invalid API name: '{}', a valid identifier may only contain letters,"
            " numbers, underscores and not starting with a number.".format(api_name)
        )

    if api_name in BENTOML_RESERVED_API_NAMES:
        raise InvalidArgument(
            "Reserved API name: '{}' is reserved for infra endpoints".format(api_name)
        )


def validate_inference_api_route(route: str):
    if re.findall(
        r"[?#]+|^(//)|^:", route
    ):  # contains '?' or '#' OR  start with '//' OR start with ':'
        # https://tools.ietf.org/html/rfc3986#page-22
        raise InvalidArgument(
            "The path {} contains illegal url characters".format(route)
        )
    if route in BENTOML_RESERVED_API_NAMES:
        raise InvalidArgument(
            "Reserved API route: '{}' is reserved for infra endpoints".format(route)
        )


class Service:
    """
    bentoml.Service is the base unit for running and deploying machine-learning models
    with BentoML. It describes how to setup the models, what are inference APIs
    available, what are their expected input output data types, and how the Service can
    be bundled for deploying elsewhere.
    """

    def __init__(self):
        self.setup()

    def setup(self):
        """
        callback function for defining the initialization process of a Service instance
        """
        pass

    @property
    def name(self):
        """
        :return: Service name
        """
        return self.__class__.name

    @property
    def version(self):
        pass

    @property
    def tag(self):
        """
        Bento tag is simply putting its name and version together, separated by a colon
        `tag` is mostly used in Yatai model management related APIs and operations
        """
        return f"{self.name}:{self.version}"

    @property
    def apis(self) -> List[InferenceAPI]:
        """
        Returns:
            list(InferenceAPI): List of Inference API objects
        """
        return self.__inference_apis

    def api(self, api_name: str):
        """
        :param api_name: the target Inference API's name
        :return:
            InferenceAPI
        """
        try:
            return next((api for api in self.__inference_apis if api.name == api_name))
        except StopIteration:
            raise NotFound(
                "Can not find API '{}' in service '{}'".format(api_name, self.name)
            )
