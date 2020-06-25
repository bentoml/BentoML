import logging
import json

import requests

from e2e_tests.docker_server.DockerTestService import DockerTestService
from e2e_tests.docker_server.utils import bento_docker_server, PORT, wait_till_server_up

PREDICT_URL = 'http://127.0.0.1:' + str(PORT) + '/predict'
PIP_CHECK_URL = 'http://127.0.0.1:' + str(PORT) + '/check_packages'
logger = logging.getLogger('bentoml.test')
save_dir = '/home/jjmachan/bentoml/test_dir'


def test_docker_container_init():
    svc = DockerTestService()
    # TODO: detemine which is better save_to_dir or save
    #path = svc.save()
    svc.save_to_dir(save_dir)
    path = save_dir
    logger.info('saving to ' + str(path))
    tag = f'{svc.name}:{svc.version}'.lower()

    with bento_docker_server(path=path, tag=tag):
        wait_till_server_up()
        r = requests.post(PREDICT_URL, json={})
        assert r.text == '"ok"', "response from server doesn't match"

        r = requests.post(PIP_CHECK_URL, json={})
        all_pass = True
        checks = json.loads(r.json())
        logger.info(checks)
        for check_item in checks.keys():
            if checks[check_item] is False:
                logger.error(check_item + ' check failed!')
                all_pass = False

        assert all_pass is True, 'Some checks failed!'
        if all_pass:
            logger.info('All systems go!')
