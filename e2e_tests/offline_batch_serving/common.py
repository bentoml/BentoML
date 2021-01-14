import logging
import os
import subprocess

logger = logging.getLogger('bentoml.test')

# load local file


# generic run predict
def delete_deployment(deployment_type, deployment_name, deployment_namespace=None):
    logger.info(f'Delete deployment {deployment_name} with BentoML CLI')
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

# testing bentoml run <bento> predict --input
def run_predict_input():
    pass

# testing bentoml run <bento> predict --input


# testing bentoml run <bento> predict --input
