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

import json
from dataclasses import asdict
from dataclasses import fields as get_fields
from dataclasses import is_dataclass


class DataclassJsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):  # pylint: disable=method-hidden
        if is_dataclass(o):
            if hasattr(o, 'to_json'):
                return o.to_json()
            else:
                return asdict(o)
        return super().default(o)


class json_serializer:
    def __init__(self, fields=None, compat=False):
        self.fields = fields
        self.compat = compat

    @staticmethod
    def _extract_nested(obj):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        return obj

    def __call__(self, klass):
        if not is_dataclass(klass):
            raise TypeError(
                f'{self.__class__.__name__} only accepts dataclasses, '
                f'got {klass.__name__}'
            )
        default_map = {
            f.name: f.default_factory() if callable(f.default_factory) else f.default
            for f in get_fields(klass)
        }
        if self.fields is None:
            self.fields = tuple(k for k in default_map.keys() if not k.startswith('_'))

        if self.compat:

            def to_json(data_obj):
                return {
                    k: self._extract_nested(getattr(data_obj, k))
                    for k in self.fields
                    if default_map[k] != getattr(data_obj, k)
                }

        else:

            def to_json(data_obj):
                return {
                    k: self._extract_nested(getattr(data_obj, k)) for k in self.fields
                }

        klass.to_json = to_json
        return klass
