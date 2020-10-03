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

import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_gcs_url(url):
    """
    Check if the url is a gcs url
    'gs://' is the standard way for Google Cloud URI
    Source: https://cloud.google.com/storage/docs/gsutil
    """
    try:
        return urlparse(url).scheme in ["gs"]
    except ValueError:
        return False
