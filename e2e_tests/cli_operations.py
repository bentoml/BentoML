import logging
import os
import subprocess

logger = logging.getLogger('bentoml.test')


def delete_deployment(deployment_type, deployment_name):
    logger.info(f'Delete deployment {deployment_name} with BentoML CLI')
    delete_deployment_command = [
        'bentoml',
        deployment_type,
        'delete',
        deployment_name,
        '--force',
    ]
    logger.info(f'Delete command: {delete_deployment_command}')
    with subprocess.Popen(
        delete_deployment_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    ) as proc:
        delete_deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info(delete_deployment_stdout)


def delete_bento(bento_name):
    logger.info(f'Deleting bento service {bento_name}')
    delete_bento_command = ['bentoml', 'delete', bento_name, '-y']
    with subprocess.Popen(
        delete_bento_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    ) as proc:
        delete_bento_stdout = proc.stdout.read().decode('utf-8')
    logger.info(delete_bento_stdout)
