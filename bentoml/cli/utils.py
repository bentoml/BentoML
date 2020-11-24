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
import sys
import threading
import itertools
import time
import logging
from datetime import datetime

import humanfriendly
from tabulate import tabulate

from bentoml.cli.click_utils import _echo
from bentoml.utils import pb_to_yaml
from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)


class Spinner:
    def __init__(self, message, delay=0.1):
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.busy = False
        self._screen_lock = None
        self.thread = None
        self.spinner_visible = False
        sys.stdout.write(message)

    def write_next(self):
        with self._screen_lock:
            if not self.spinner_visible:
                sys.stdout.write(next(self.spinner))
                self.spinner_visible = True
                sys.stdout.flush()

    def remove_spinner(self, cleanup=False):
        with self._screen_lock:
            if self.spinner_visible:
                sys.stdout.write('\b')
                self.spinner_visible = False
                if cleanup:
                    sys.stdout.write(' ')  # overwrite spinner with blank
                    sys.stdout.write('\r')  # move to next line
                sys.stdout.flush()

    def spinner_task(self):
        while self.busy:
            self.write_next()
            time.sleep(self.delay)
            self.remove_spinner()

    def __enter__(self):
        if sys.stdout.isatty():
            self._screen_lock = threading.Lock()
            self.busy = True
            self.thread = threading.Thread(target=self.spinner_task)
            self.thread.start()

    def __exit__(self, exception, value, tb):
        if sys.stdout.isatty():
            self.busy = False
            self.remove_spinner(cleanup=True)
        else:
            sys.stdout.write('\r')


def parse_key_value_pairs(key_value_pairs_str):
    result = {}
    if key_value_pairs_str:
        for key_value_pair in key_value_pairs_str.split(','):
            key, value = key_value_pair.split('=')
            key = key.strip()
            value = value.strip()
            if key in result:
                logger.warning("duplicated key '%s' found string map parameter", key)
            result[key] = value
    return result


def echo_docker_api_result(docker_generator):
    layers = {}
    for line in docker_generator:
        if "stream" in line:
            cleaned = line['stream'].rstrip("\n")
            if cleaned != "":
                yield cleaned
        if "status" in line and line["status"] == "Pushing":
            progress = line["progressDetail"]
            layers[line["id"]] = progress["current"], progress["total"]
            cur, total = zip(*layers.values())
            cur, total = (
                humanfriendly.format_size(sum(cur)),
                humanfriendly.format_size(sum(total)),
            )
            yield f"Pushed {cur} / {total}"
        if "errorDetail" in line:
            error = line["errorDetail"]
            raise BentoMLException(error["message"])


def _print_deployment_info(deployment, output_type):
    if output_type == 'yaml':
        _echo(pb_to_yaml(deployment))
    else:
        from google.protobuf.json_format import MessageToDict

        deployment_info = MessageToDict(deployment)
        if deployment_info['state'] and deployment_info['state']['infoJson']:
            deployment_info['state']['infoJson'] = json.loads(
                deployment_info['state']['infoJson']
            )
        _echo(json.dumps(deployment_info, indent=2, separators=(',', ': ')))


def _format_labels_for_print(labels):
    if not labels:
        return None
    result = [f'{label_key}:{labels[label_key]}' for label_key in labels]
    return '\n'.join(result)


def _format_deployment_age_for_print(deployment_pb):
    if not deployment_pb.created_at:
        # deployments created before version 0.4.5 don't have created_at field,
        # we will not show the age for those deployments
        return None
    else:
        return human_friendly_age_from_datetime(deployment_pb.created_at.ToDatetime())


def human_friendly_age_from_datetime(dt, detailed=False, max_unit=2):
    return humanfriendly.format_timespan(datetime.utcnow() - dt, detailed, max_unit)


def _print_deployments_table(deployments, wide=False):
    from bentoml.yatai.proto.deployment_pb2 import DeploymentState, DeploymentSpec

    table = []
    if wide:
        headers = [
            'NAME',
            'NAMESPACE',
            'PLATFORM',
            'BENTO_SERVICE',
            'STATUS',
            'AGE',
            'LABELS',
        ]
    else:
        headers = [
            'NAME',
            'NAMESPACE',
            'PLATFORM',
            'BENTO_SERVICE',
            'STATUS',
            'AGE',
        ]
    for deployment in deployments:
        row = [
            deployment.name,
            deployment.namespace,
            DeploymentSpec.DeploymentOperator.Name(deployment.spec.operator)
            .lower()
            .replace('_', '-'),
            f'{deployment.spec.bento_name}:{deployment.spec.bento_version}',
            DeploymentState.State.Name(deployment.state.state)
            .lower()
            .replace('_', ' '),
            _format_deployment_age_for_print(deployment),
        ]
        if wide:
            row.append(_format_labels_for_print(deployment.labels))
        table.append(row)
    table_display = tabulate(table, headers, tablefmt='plain')
    _echo(table_display)


def _print_deployments_info(deployments, output_type):
    if output_type == 'table':
        _print_deployments_table(deployments)
    elif output_type == 'wide':
        _print_deployments_table(deployments, wide=True)
    else:
        for deployment in deployments:
            _print_deployment_info(deployment, output_type)
