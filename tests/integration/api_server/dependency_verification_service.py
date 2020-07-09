import bentoml
from bentoml.saved_bundle import save_to_dir
from bentoml.adapters import JsonInput


@bentoml.env(pip_dependencies=['rich'])
class DependencyVerificationService(bentoml.BentoService):
    """
    This service checks all the dependencies and verifies everything
    is setup properly
    """

    @bentoml.api(input=JsonInput())
    def test_packages(self, json):
        print(json)


def gen_test_bundle(tmpdir):
    # When the ExampleBentoService got saved and loaded again in the test, the two class
    # attribute below got set to the loaded BentoService class. Resetting it here so it
    # does not effect other tests
    DependencyVerificationService._bento_service_bundle_path = None
    DependencyVerificationService._bento_service_bundle_version = None
    test_svc = DependencyVerificationService()

    save_to_dir(test_svc, tmpdir, silent=True)
