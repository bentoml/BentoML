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

import os
import azure.functions as func  # pylint: disable=import-error, no-name-in-module

from bentoml.server.api_server import BentoAPIServer
from bentoml.saved_bundle import load_from_dir

bento_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
svc = load_from_dir(bento_path)

bento_server = BentoAPIServer(svc)


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return func.WsgiMiddleware(bento_server.app.wsgi_app).handle(req, context)
