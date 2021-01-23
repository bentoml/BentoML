import logging
import os
import subprocess

logger = logging.getLogger('bentoml.test')

# load local file helper


# generic run predict
def run_predict(bento, input_data, is_file=False):
    run_predict_deployment_command = [
        'bentoml',
        'run',
        bento,
        'predict',
        '--input-file' if is_file else '--input',
        input_data,
    ]

    logger.info(f'Run predict command: {run_predict_deployment_command}')
    with subprocess.Popen(
        run_predict_deployment_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ,
    ) as proc:
        run_predict_deployment_stdout = proc.stdout.read().decode('utf-8')
    logger.info(f'Got output: {run_predict_deployment_stdout}')
    return run_predict_deployment_stdout


# testing bentoml run <bento> predict --input
def run_predict_input():
    pass


# testing bentoml run <bento> predict --input-file
