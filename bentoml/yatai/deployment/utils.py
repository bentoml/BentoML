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

from bentoml.exceptions import InvalidArgument


def raise_if_api_names_not_found_in_bento_service_metadata(metadata, api_names):
    all_api_names = [api.name for api in metadata.apis]

    if not set(api_names).issubset(all_api_names):
        raise InvalidArgument(
            "Expect api names {api_names} to be "
            "subset of {all_api_names}".format(
                api_names=api_names, all_api_names=all_api_names
            )
        )
