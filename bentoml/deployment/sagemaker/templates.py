# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
"""
