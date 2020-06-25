import os
import logging
import json

from bentoml import BentoService, api, env
from bentoml.adapters import JsonInput

logger = logging.getLogger('bentoml.test')

@env(
        pip_dependencies=['scikit-learn'],
        setup_sh='touch success.file',
    )
class DockerTestService(BentoService):
    @api(input=JsonInput())
    def predict(self, data):
        logger.info(f'DockerTestService predict API received data {data}')
        return 'ok'

    @api(input=JsonInput())
    def check_packages(self, data):
        logger.info('testing the packages installed...')
        checks = {
                    'pip-install': True,
                    'setup-script': True,
                    'conda-install': True,
                }
        try:
            import sklearn
        except:
            checks['pip-install'] = False

        if not os.path.isfile('/bento/success.file'):
            checks['setup-script'] = False

        checks = json.dumps(checks)
        return checks
