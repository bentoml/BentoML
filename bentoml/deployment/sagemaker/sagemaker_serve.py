#!/usr/bin/env python

# This implement the sagemaker serving service shell.  It starts nginx and gunicorn.
# Parameter               Env Var                      Default Value
# number of workers       BENTO_SERVER_TIMEOUT         60s
# timeout                 GUNICORN_WORKER_COUNT        number of cpu cores / 2 + 1
# api name                API_NAME                     None

import subprocess
import os
import signal
import sys

from bentoml.server.utils import get_gunicorn_num_of_workers

bento_server_timeout = os.environ.get('BENTO_SERVER_TIMEOUT', 60)
bento_server_workers = int(
    os.environ.get('GUNICORN_WORKER_COUNT', get_gunicorn_num_of_workers())
)


def sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)


def _serve():
    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(['ln', '-sf', '/dev/stdout', '/var/log/nginx/access.log'])
    subprocess.check_call(['ln', '-sf', '/dev/stderr', '/var/log/nginx/error.log'])

    nginx = subprocess.Popen(['nginx', '-c', '/opt/program/nginx.conf'])
    gunicorn_app = subprocess.Popen(
        [
            'gunicorn',
            '--timeout',
            str(bento_server_timeout),
            '-k',
            'gevent',
            '-b',
            'unix:/tmp/gunicorn.sock',
            '-w',
            str(bento_server_workers),
            'wsgi:app',
        ]
    )
    signal.signal(
        signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn_app.pid)
    )

    pids = set([nginx.pid, gunicorn_app.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break
    sigterm_handler(nginx.pid, gunicorn_app.pid)
    print('Inference server exiting')


if __name__ == '__main__':
    _serve()
