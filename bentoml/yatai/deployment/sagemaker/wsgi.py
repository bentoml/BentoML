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

from bentoml.saved_bundle import load_from_dir
from bentoml.yatai.deployment.sagemaker.model_server import BentomlSagemakerServer

api_name = os.environ.get('API_NAME', None)
model_service = load_from_dir('/bento')
server = BentomlSagemakerServer(model_service, api_name)
app = server.app
