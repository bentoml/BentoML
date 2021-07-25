import typing as t

from simple_di import Provide, inject

from .._internal.configuration.containers import BentoMLContainer


# file structure
#
# $BENTOML_HOME/model_store/{defined_name}/{version}/artifact.save() content
#
# path to parse for artifact.save: $BENTOML_HOME/model_store/{defined_name}/{versions}
class _ModelManager:
    """
    Manages versions of saved artifacts for local development workflow
    and online serving within BentoBundle
    """

    @inject
    def __init__(
        self, home_dir: t.Optional[str] = Provide[BentoMLContainer.bentoml_home]
    ):
        self.bento_home = home_dir
