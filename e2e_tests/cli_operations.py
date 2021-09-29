import logging
import os
import subprocess

logger = logging.getLogger('bentoml.test')


def delete_deployment(deployment_type, deployment_name, deployment_namespace=None):
    logger.info(f'Delete deployment {deployment_name} with Qwak CLI')
    delete_deployment_command = [
        'bentoml',
        deployment_type,
        'delete',
        deployment_name,
        '--force',
    ]
    if deployment_namespace:
        delete_deployment_command.extend(['--namespace', deployment_namespace])
    logger.info(f'Delete command: {delete_deployment_command}')
    with subprocess.Popen(
        delete_deployment_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    ) as proc:
        delete_deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info(delete_deployment_stdout)
    return delete_deployment_stdout


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
    return delete_bento_stdout
