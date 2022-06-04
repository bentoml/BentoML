import os
import shutil
import logging
import importlib
import importlib.util

from ..configuration import is_pypi_installed_bentoml
from ..utils.tempdir import TempDirectory
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)

BENTOML_DEV_BUILD = "BENTOML_BUNDLE_LOCAL_BUILD"


def build_bentoml_editable_wheel(target_path: str) -> None:
    """
    This is for BentoML developers to create Bentos that contains the local bentoml
    build based on their development branch. To enable this behavior, one must
    set envar :code:`BENTOML_BUNDLE_LOCAL_BUILD=True` before building a Bento.

    Args:
        target_path(:code:`str`):
            The path to the wheel file to be created.

    .. note::
        If bentoml is installed in editor mode(:code:`pip install -e`), this will build a wheel
        distribution with the local bentoml source and add it to saved bento directory
        under :code:`{bento_path}/env/docker/whl/`
    """
    if str(os.environ.get(BENTOML_DEV_BUILD, False)).lower() != "true":
        return

    if is_pypi_installed_bentoml():
        # skip this entirely if BentoML is installed from PyPI
        return

    # Find bentoml module path
    (module_location,) = importlib.util.find_spec("bentoml").submodule_search_locations  # type: ignore # noqa

    # PEP517-compatible
    bentoml_pyproject = os.path.abspath(os.path.join(module_location, "..", "pyproject.toml"))  # type: ignore

    # this is for BentoML developer to create Service containing custom development
    # branches of BentoML library, it is True only when BentoML module is installed
    # in development mode via "pip install --editable ."
    if os.path.isfile(bentoml_pyproject):
        logger.info(
            "BentoML is installed in `editable` mode; building BentoML distribution with the local BentoML code base. The built wheel file will be included in the target bento. The wheel is also PEP517-compatible."
        )
        try:
            from build import ProjectBuilder
        except ModuleNotFoundError:
            raise BentoMLException(
                f'{BENTOML_DEV_BUILD} is set to True, but `build` is not installed. Make sure to do `pip install -r requirements/dev-requirements.txt` and try again.'
            )

        with TempDirectory() as dist_dir:
            builder = ProjectBuilder(os.path.dirname(bentoml_pyproject))
            builder.build("wheel", dist_dir)  # type: ignore (incomplete TempDirectory stub)
            shutil.copytree(dist_dir, target_path)  # type: ignore (incomplete TempDirectory stub)
    else:
        logger.info(
            "Custom BentoML build is detected. For a Bento to use the same build at serving time, add your custom BentoML build to the pip packages list, e.g. `packages=['git+https://github.com/bentoml/bentoml.git@13dfb36']`"
        )
