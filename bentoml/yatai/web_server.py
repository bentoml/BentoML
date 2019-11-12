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

import os

from werkzeug.middleware.shared_data import SharedDataMiddleware
from flask import Flask, send_from_directory, current_app, request, Response

from bentoml.yatai import get_yatai_service

static_dir = 'web_server_front_end/build'

app = Flask(__name__)
# Using the shared data middleware we can let the Flask app serve everything,
# but you need to do a `npm run build` for it which creates a production-like
# build.  For an actual production deployment you would have your web server
# serve the data from `.../client/build/` and just forward API requests to
# the Flask app!
app.wsgi_app = SharedDataMiddleware(
    app.wsgi_app,
    {'/': os.path.join(os.path.dirname(__file__), 'web_server_front_end', 'build')},
)

static_dir_path = os.path.join(app.root_path, static_dir)
yatai_service = get_yatai_service()

# Serve the index.html for the React App for all other routes.
@app.route('/')
def index():
    print(os.path.join(current_app.root_path, 'web_server_front_end', 'build'))
    return send_from_directory(
        os.path.join(current_app.root_path, 'web_server_front_end', 'build'),
        'index.html',
    )


@app.route('/create_deployment', methods=['POST'])
def create_deployment():
    print(request)
    return Response(status=200)


@app.route('/apply_deployment', methods=['POST'])
def apply_deployment():
    pass


@app.route('/get_deployment')
def get_deployment():
    pass


@app.route('/delete_deployment', methods=['DELETE'])
def delete_deployment():
    pass


@app.route('/list_deployments')
def list_deployments():
    pass


def start_web_server(port=5000):
    app.run(port=port)
