import importlib
import importlib.util
import logging
import os
import shutil

from ..configuration import is_pypi_installed_bentoml
from ..utils.tempdir import TempDirectory

logger = logging.getLogger(__name__)


def build_bentoml_whl_to_target_if_in_editable_mode(target_path):
    """
    if bentoml is installed in editor mode(pip install -e), this will build a wheel
    distribution with the local bentoml source and add it to saved bento directory
    under {bento_path}/env/python/wheels/
    """
    if is_pypi_installed_bentoml():
        # skip this entirely if BentoML is installed from PyPI
        return

    # Find bentoml module path
    (module_location,) = importlib.util.find_spec("bentoml").submodule_search_locations  # type: ignore # noqa

    bentoml_setup_py = os.path.abspath(os.path.join(module_location, "..", "setup.py"))

    # this is for BentoML developer to create Service containing custom development
    # branches of BentoML library, it is True only when BentoML module is installed
    # in development mode via "pip install --editable ."
    if os.path.isfile(bentoml_setup_py):
        logger.warning(
            "BentoML is installed in `editable` model, building BentoML distribution with local BentoML code base. The built wheel file will be included in target bento directory, under: {bento_path}/env/python/wheels/"
        )

        with TempDirectory() as tempdir:
            from setuptools import sandbox

            # build BentoML wheel distribution under tempdir
            sandbox.run_setup(
                bentoml_setup_py,
                ["-q", "bdist_wheel", "--dist-dir", tempdir],
            )

            # copy the built wheel file to target directory
            shutil.copytree(tempdir, target_path)
