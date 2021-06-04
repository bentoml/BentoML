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

import argparse
import functools
import inspect
import itertools
import logging
import sys
from typing import Callable, Iterable, Iterator, Sequence

from bentoml.adapters import BaseInputAdapter, BaseOutputAdapter
from bentoml.exceptions import BentoMLConfigException
from bentoml.tracing import get_tracer
from bentoml.types import HTTPRequest, HTTPResponse, InferenceResult, InferenceTask
from bentoml.utils import cached_property


logger = logging.getLogger(__name__)
prediction_logger = logging.getLogger("bentoml.prediction")


class InferenceAPI(object):
    """
    InferenceAPI defines an inference call to the underlying model, including its input
    and output adapter, the user-defined API callback function, and configurations for
    working with the BentoML adaptive micro-batching mechanism
    """

    def __init__(
        self,
        service,
        name,
        doc,
        input_adapter: BaseInputAdapter,
        user_func: Callable,
        output_adapter: BaseOutputAdapter,
        mb_max_latency=10000,
        mb_max_batch_size=1000,
        batch=False,
        route=None,
    ):
        """
        :param service: ref to service containing this API
        :param name: API name
        :param doc: the user facing document of this inference API, default to the
            docstring of the inference API function
        :param input_adapter: A InputAdapter that transforms HTTP Request and/or
            CLI options into parameters for API func
        :param user_func: the user-defined API callback function, this is
            typically the 'predict' method on a model
        :param output_adapter: A OutputAdapter is an layer between result of user
            defined API callback function
            and final output in a variety of different forms,
            such as HTTP response, command line stdout or AWS Lambda event object.
        :param mb_max_latency: The latency goal of this inference API in milliseconds.
            Default: 10000.
        :param mb_max_batch_size: The maximum size of requests batch accepted by this
            inference API. This parameter governs the throughput/latency trade off, and
            avoids having large batches that exceed some resource constraint (e.g. GPU
            memory to hold the entire batch's data). Default: 1000.
        :param batch: If true, the user API functool would take a batch of input data
            a time.
        :param route: Specify HTTP URL route of this inference API. By default,
            `api.name` is used as the route.  This parameter can be used for customizing
            the URL route, e.g. `route="/api/v2/model_a/predict"`
        """
        self._service = service
        self._name = name
        self._input_adapter = input_adapter
        self._user_func = user_func
        self._output_adapter = output_adapter
        self.mb_max_latency = mb_max_latency
        self.mb_max_batch_size = mb_max_batch_size
        self.batch = batch
        self.route = name if route is None else route

        if not self.input_adapter.BATCH_MODE_SUPPORTED and batch:
            raise BentoMLConfigException(
                f"{input_adapter.__class__.__name__} does not support `batch=True`"
            )

        if not self.input_adapter.SINGLE_MODE_SUPPORTED and not batch:
            raise BentoMLConfigException(
                f"{input_adapter.__class__.__name__} does not support `batch=False`, "
                "its output passed to API functions could only be a batch of data."
            )

        if doc is None:
            # generate a default doc string for this inference API
            doc = (
                f"BentoService inference API '{self.name}', input: "
                f"'{type(input_adapter).__name__}', output: "
                f"'{type(output_adapter).__name__}'"
            )
        self._doc = doc

    @property
    def service(self):
        """
        :return: a reference to the BentoService serving this inference API
        """
        return self._service

    @property
    def name(self):
        """
        :return: the name of this inference API
        """
        return self._name

    @property
    def doc(self):
        """
        :return: user facing documentation of this inference API
        """
        return self._doc

    @property
    def input_adapter(self) -> BaseInputAdapter:
        """
        :return: the input adapter of this inference API
        """
        return self._input_adapter

    @property
    def output_adapter(self) -> BaseOutputAdapter:
        """
        :return: the output adapter of this inference API
        """
        return self._output_adapter

    @cached_property
    def user_func(self):
        """
        :return: user-defined inference API callback function
        """

        # allow user to define handlers without 'tasks' kwargs
        _sig = inspect.signature(self._user_func)
        if self.batch:
            append_arg = "tasks"
        else:
            append_arg = "task"
        try:
            _sig.bind_partial(**{append_arg: None})
            append_arg = None
        except TypeError:
            pass

        @functools.wraps(self._user_func)
        def wrapped_func(*args, **kwargs):
            with get_tracer().span(
                service_name=f"BentoService.{self.service.name}",
                span_name=f"InferenceAPI {self.name} user defined callback function",
            ):
                if append_arg and append_arg in kwargs:
                    tasks = kwargs.pop(append_arg)
                elif append_arg in kwargs:
                    tasks = kwargs[append_arg]
                else:
                    tasks = []
                try:
                    return self._user_func(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Error caught in API function:", exc_info=1)
                    if self.batch:
                        for task in tasks:
                            if not task.is_discarded:
                                task.discard(
                                    http_status=500,
                                    err_msg=f"Exception happened in API function: {e}",
                                )
                        return [None] * sum(
                            1 if t.batch is None else t.batch for t in tasks
                        )
                    else:
                        task = tasks
                        if not task.is_discarded:
                            task.discard(
                                http_status=500,
                                err_msg=f"Exception happened in API function: {e}",
                            )
                        return [None] * (1 if task.batch is None else task.batch)

        return wrapped_func

    @property
    def request_schema(self):
        """
        :return: the HTTP API request schema in OpenAPI/Swagger format
        """
        if self.input_adapter.custom_request_schema is None:
            schema = self.input_adapter.request_schema
        else:
            schema = self.input_adapter.custom_request_schema

        if schema.get('application/json'):
            schema.get('application/json')[
                'example'
            ] = self.input_adapter._http_input_example
        return schema

    def _filter_tasks(
        self, inf_tasks: Iterable[InferenceTask]
    ) -> Iterator[InferenceTask]:
        for task in inf_tasks:
            if task.is_discarded:
                continue
            try:
                self.input_adapter.validate_task(task)
                yield task
            except AssertionError as e:
                task.discard(http_status=400, err_msg=str(e))

    def infer(self, inf_tasks: Iterable[InferenceTask]) -> Sequence[InferenceResult]:
        inf_tasks = tuple(inf_tasks)

        # extract args
        user_args = self.input_adapter.extract_user_func_args(inf_tasks)
        filtered_tasks = tuple(t for t in inf_tasks if not t.is_discarded)

        # call user function
        if not self.batch:  # For single inputs
            user_return = []
            for task, legacy_user_args in zip(
                filtered_tasks,
                self.input_adapter.iter_batch_args(user_args, tasks=filtered_tasks),
            ):
                ret = self.user_func(*legacy_user_args, task=task)
                if task.is_discarded:
                    continue
                else:
                    user_return.append(ret)
            if (
                isinstance(user_return, (list, tuple))
                and len(user_return)
                and isinstance(user_return[0], InferenceResult)
            ):
                inf_results = user_return
            else:
                # pack return value
                filtered_tasks = tuple(t for t in inf_tasks if not t.is_discarded)
                inf_results = self.output_adapter.pack_user_func_return_value(
                    user_return, tasks=filtered_tasks
                )
        else:
            user_return = self.user_func(*user_args, tasks=filtered_tasks)
            if (
                isinstance(user_return, (list, tuple))
                and len(user_return)
                and isinstance(user_return[0], InferenceResult)
            ):
                inf_results = user_return
            else:
                # pack return value
                filtered_tasks = tuple(t for t in inf_tasks if not t.is_discarded)
                inf_results = self.output_adapter.pack_user_func_return_value(
                    user_return, tasks=filtered_tasks
                )

        full_results = InferenceResult.complete_discarded(inf_tasks, inf_results)

        log_data = dict(
            service_name=self.service.name if self.service else "",
            service_version=self.service.version if self.service else "",
            api=self.name,
        )
        for task, result in zip(inf_tasks, inf_results):
            prediction_logger.info(
                dict(
                    log_data,
                    task=task.to_json(),
                    result=result.to_json(),
                    request_id=task.task_id,
                )
            )

        return tuple(full_results)

    def handle_request(self, request: HTTPRequest) -> HTTPResponse:
        inf_task = self.input_adapter.from_http_request(request)
        results = self.infer((inf_task,))
        result = next(iter(results))
        response = self.output_adapter.to_http_response(result)
        response.headers['X-Request-Id'] = inf_task.task_id
        return response

    def handle_batch_request(self, requests: Sequence[HTTPRequest]):
        with get_tracer().span(
            service_name=f"BentoService.{self.service.name}",
            span_name=f"InferenceAPI {self.name} handle batch requests",
        ):
            inf_tasks = tuple(map(self.input_adapter.from_http_request, requests))
            results = self.infer(inf_tasks)
            responses = tuple(map(self.output_adapter.to_http_response, results))
            for inf_task, response in zip(inf_tasks, responses):
                response.headers['X-Request-Id'] = inf_task.task_id
            return responses

    def handle_cli(self, cli_args: Sequence[str]) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--max-batch-size", default=sys.maxsize, type=int)
        parsed_args, _ = parser.parse_known_args(cli_args)

        exit_code = 0

        tasks_iter = self.input_adapter.from_cli(tuple(cli_args))
        while True:
            tasks = tuple(itertools.islice(tasks_iter, parsed_args.max_batch_size))
            if not len(tasks):
                break
            results = self.infer(tasks)
            exit_code = exit_code or self.output_adapter.to_cli(results)

        return exit_code

    def handle_aws_lambda_event(self, event):
        inf_task = self.input_adapter.from_aws_lambda_event(event)
        result = next(iter(self.infer((inf_task,))))
        out_event = self.output_adapter.to_aws_lambda_event(result)
        if isinstance(out_event, dict) and "headers" in out_event:
            headers = out_event.get("headers", dict())
            headers["Access-Control-Allow-Origin"] = "*"  # TODO: make it configurable
            out_event["headers"] = headers
        return out_event
