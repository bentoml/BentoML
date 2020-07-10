import json
import logging

import bentoml
from bentoml.saved_bundle import save_to_dir
from bentoml.adapters import JsonInput

logger = logging.getLogger("bentoml")


@bentoml.env(
    pip_dependencies=['rich'],
    conda_dependencies=['scikit-learn'],
    # requirements_txt_file='tests/integration/api_server/requirements.txt',
)
class DependencyVerificationService(bentoml.BentoService):
    """
    This service checks all the dependencies and verifies everything
    is setup properly
    """

    @bentoml.api(input=JsonInput())
    def test_packages(self, json_input):
        is_installed = {
            'rich': True,
            'sklearn': True,
            'pillow': True,
        }

        try:
            import rich  # NOQA pylint: disable=unused-import
        except ImportError:
            is_installed['rich'] = False

        try:
            import sklearn  # NOQA pylint: disable=unused-import
        except ImportError:
            is_installed['sklearn'] = False

        try:
            import PIL  # NOQA pylint: disable=unused-import
        except ImportError:
            is_installed['pillow'] = False

        logger.info(json_input, is_installed)
        out = json.dumps(is_installed)
        logger.info(out)
        return [out]


def gen_test_bundle(tmpdir):
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    DependencyVerificationService._bento_service_bundle_path = None
    DependencyVerificationService._bento_service_bundle_version = None
    test_svc = DependencyVerificationService()

    save_to_dir(test_svc, tmpdir, silent=True)
