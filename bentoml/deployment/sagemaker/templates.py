# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

DEFAULT_NGINX_CONFIG = """\
worker_processes 1;
daemon off; # Prevent forking

pid /tmp/nginx.pid;
error_log /var/log/nginx/error.log;

events {
  # defaults
}

http {
  include /etc/nginx/mime.types;
  default_type application/octet-stream;
  access_log /var/log/nginx/access.log combined;

  upstream gunicorn {
    server unix:/tmp/gunicorn.sock;
  }

  server {
    listen 8080 deferred;
    client_max_body_size 500m;

    keepalive_timeout 5;
    proxy_read_timeout 1200s;

    location ~ ^/(ping|invocations) {
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header Host $http_host;
      proxy_redirect off;
      proxy_pass http://gunicorn;
    }

    location / {
      return 404 "{}";
    }
  }
}
"""

DEFAULT_WSGI_PY = """\
import os

from bentoml.archive import load
from bentoml.server.bento_sagemaker_server import BentoSagemakerServer

api_name = os.environ.get('API_NAME', None)
model_service = load('/opt/program')
server = BentoSagemakerServer(model_service, api_name)
app = server.app
"""

DEFAULT_SERVE_SCRIPT = """\
#!/usr/bin/env python

# This implement the sagemaker serving service shell.  It starts nginx and gunicorn.
# Parameter                    Env Var                          Default Value
# number of workers            MODEL_SERVER_TIMEOUT             60s
# timeout                      MODEL_SERVER_WORKERS             number of cpu cores / 2 + 1
# api name                     API_NAME                         None

import subprocess
import os
import signal
import sys

from bentoml.archive import load
from bentoml.server.bento_sagemaker_server import BentoSagemakerServer
from bentoml.server.gunicorn_server import GunicornApplication, get_gunicorn_worker_count


model_server_timeout = os.environ.get('MODEL_SERVER_TIMEOUT', 60)
model_server_workers = int(os.environ.get('MODEL_SERVER_WORKERS', get_gunicorn_worker_count()))


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
    gunicorn_app = subprocess.Popen(['gunicorn',
                                     '--timeout', str(model_server_timeout),
                                     '-k', 'gevent',
                                     '-b', 'unix:/tmp/gunicorn.sock',
                                     '-w', str(model_server_workers),
                                     'wsgi:app'])
    signal.signal(signal.SIGTERM, lambda a, b: sigterm_handler(nginx.pid, gunicorn_app.pid))

    pids = set([nginx.pid, gunicorn_app.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break
    sigterm_handler(nginx.pid, gunicorn_app.pid)
    print('Inference server exiting')


if __name__ == '__main__':
  _serve()
"""  # noqa: E501
