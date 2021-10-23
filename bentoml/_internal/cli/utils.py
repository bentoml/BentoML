import itertools
import logging
import sys
import threading
import time
from datetime import datetime

import humanfriendly

from bentoml.exceptions import BentoMLException

logger = logging.getLogger(__name__)


class Spinner:
    def __init__(self, message, delay=0.1):
        self.spinner = itertools.cycle(["-", "/", "|", "\\"])
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
                sys.stdout.write("\b")
                self.spinner_visible = False
                if cleanup:
                    sys.stdout.write(" ")  # overwrite spinner with blank
                    sys.stdout.write("\r")  # move to next line
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
            sys.stdout.write("\r")


def parse_key_value_pairs(key_value_pairs_str):
    result = {}
    if key_value_pairs_str:
        for key_value_pair in key_value_pairs_str.split(","):
            key, value = key_value_pair.split("=")
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
            cleaned = line["stream"].rstrip("\n")
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


def _format_labels_for_print(labels):
    if not labels:
        return None
    result = [f"{label_key}:{labels[label_key]}" for label_key in labels]
    return "\n".join(result)


def _format_deployment_age_for_print(deployment_pb):
    if not deployment_pb.created_at:
        # deployments created before version 0.4.5 don't have created_at field,
        # we will not show the age for those deployments
        return None
    else:
        return human_friendly_age_from_datetime(deployment_pb.created_at.ToDatetime())


def human_friendly_age_from_datetime(dt, detailed=False, max_unit=2):
    return humanfriendly.format_timespan(datetime.utcnow() - dt, detailed, max_unit)
