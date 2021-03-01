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
import itertools
import pathlib
import sys
from typing import Iterable, Iterator, NamedTuple, Sequence, Tuple

from bentoml.types import (
    ApiFuncArgs,
    AwsLambdaEvent,
    BatchApiFuncArgs,
    FileLike,
    HTTPRequest,
    InferenceTask,
)


class BaseInputAdapter:
    """
    InputAdapter is an abstraction layer between user defined API callback function
    and prediction request input in a variety of different forms, such as HTTP request
    body, command line arguments or AWS Lambda event object.
    """

    HTTP_METHODS = ["POST", "GET"]
    BATCH_MODE_SUPPORTED = True
    SINGLE_MODE_SUPPORTED = True

    def __init__(self, http_input_example=None, **base_config):
        self._config = base_config
        self._http_input_example = http_input_example
        self.custom_request_schema = base_config.get('request_schema')

    @property
    def config(self):
        return self._config

    @property
    def request_schema(self):
        """
        :return: OpenAPI json schema for the HTTP API endpoint created with this input
                 adapter
        """
        return {"application/json": {"schema": {"type": "object"}}}

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this InputAdapter
        """
        return []

    def from_http_request(self, req: HTTPRequest) -> InferenceTask:
        """
        Handles HTTP requests, convert it into InferenceTask
        """
        raise NotImplementedError()

    def from_aws_lambda_event(self, event: AwsLambdaEvent) -> InferenceTask:
        """
        Handles AWS lambda events, convert it into InferenceTask
        """
        raise NotImplementedError()

    def from_cli(self, cli_args: Tuple[str, ...]) -> Iterator[InferenceTask]:
        """
        Handles CLI command, generate InferenceTask
        """
        raise NotImplementedError()

    def from_inference_job(self, *_args, **_kwargs) -> Iterator[InferenceTask]:
        """
        Handles <bento_inference_api>.run(), generate InferenceTask
        """
        raise NotImplementedError()

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask]
    ) -> BatchApiFuncArgs:
        """
        Extract args that user API function is expecting from InferenceTask
        """
        raise NotImplementedError()

    def iter_batch_args(
        self,
        batch_args: BatchApiFuncArgs,
        tasks: InferenceTask = None,  # pylint: disable=unused-argument
    ) -> Iterator[ApiFuncArgs]:
        """
        Extract args that user API function is expecting from InferenceTask
        """
        return iter(zip(*batch_args))


COLOR_FAIL = '\033[91m'


def exit_cli(err_msg: str = "", exit_code: int = None):
    if exit_code is None:
        exit_code = 1 if err_msg else 0
    if err_msg:
        print(f"{COLOR_FAIL}{err_msg}", file=sys.stderr)
    sys.exit(exit_code)


class CliInputParser(NamedTuple):
    arg_names: Tuple[str]
    file_arg_names: Tuple[str]
    arg_strs: Tuple[str]
    file_arg_strs: Tuple[str]
    parser: argparse.ArgumentParser

    @classmethod
    @functools.lru_cache()
    def get(cls, input_names: Tuple[str] = None):
        arg_names = (
            tuple(f"input_{n}" for n in input_names) if input_names else ("input",)
        )
        arg_strs = tuple(f'--{n.replace("_", "-")}' for n in arg_names)

        file_arg_names = (
            tuple(f"input_file_{n}" for n in input_names)
            if input_names
            else ("input_file",)
        )
        file_arg_strs = tuple(f'--{n.replace("_", "-")}' for n in file_arg_names)

        parser = argparse.ArgumentParser()
        for name in itertools.chain(arg_strs, file_arg_strs):
            parser.add_argument(name, nargs="+")

        return cls(arg_names, file_arg_names, arg_strs, file_arg_strs, parser)

    def parse(self, args: Sequence[str]):
        try:
            parsed, _ = self.parser.parse_known_args(args)
        except SystemExit:
            parsed = None

        inputs = tuple(getattr(parsed, name, None) for name in self.arg_names)
        file_inputs = tuple(getattr(parsed, name, None) for name in self.file_arg_names)

        if any(inputs) and any(file_inputs):
            exit_cli(
                '''
                Conflict arguments:
                --input* and --input-file* should not be provided at same time
                '''
            )
        if not all(inputs) and not all(file_inputs):
            exit_cli(
                f'''
                Insufficient arguments:
                ({' '.join(self.arg_strs)}) or
                ({' '.join(self.file_arg_strs)})
                are required
                '''
            )

        if all(inputs):
            if functools.reduce(lambda i, j: len(i) == len(j), inputs):
                return inputs, None
            else:
                exit_cli(
                    f'''
                    Arguments length mismatch:
                    Each ({' '.join(self.arg_strs)})
                    should have same amount of inputs
                    '''
                )

        if all(file_inputs):
            if functools.reduce(lambda i, j: len(i) == len(j), file_inputs):
                return None, file_inputs
            else:
                exit_cli(
                    f'''
                    Arguments length mismatch:
                    Each ({' '.join(self.file_arg_strs)})
                    should have same amount of inputs
                    '''
                )


def parse_cli_inputs(
    args: Sequence[str], input_names: Sequence[str] = None
) -> Iterator[Tuple[FileLike]]:
    '''
    Parse CLI args and iter each pair of inputs in bytes.

    >>> parse_cli_inputs("--input-x '1' '2' --input-y 'a' 'b'".split(' '), ('x', 'y'))
    >>> parse_cli_inputs(
    >>>     "--input-file-x 1.jpg 2.jpg --input-file-y 1.label 2.label".split(' '),
    >>>     ('x', 'y'))
    '''
    parser = CliInputParser.get(tuple(input_names))
    return parser.parse(args)


def parse_cli_input(cli_args: Iterable[str]) -> Iterator[FileLike]:
    '''
    Parse CLI args and iter each input in bytes.

    >>> parse_cli_input('--input {"input":1} {"input":2}'.split(' '))
    OR
    >>> parse_cli_inputs("--input-file 1.jpg 2.jpg 3.jpg".split(' '))
    '''
    parser = argparse.ArgumentParser()
    input_g = parser.add_mutually_exclusive_group(required=True)
    input_g.add_argument('--input', nargs="+", type=str)
    input_g.add_argument('--input-file', nargs="+")

    parsed_args, _ = parser.parse_known_args(list(cli_args))

    inputs = tuple(
        parsed_args.input if parsed_args.input_file is None else parsed_args.input_file
    )
    is_file = parsed_args.input_file is not None
    if is_file:
        for input_ in inputs:
            uri = pathlib.Path(input_).absolute().as_uri()
            yield FileLike(uri=uri)

    else:
        for input_ in inputs:
            rv = FileLike(bytes_=input_.encode())
            yield rv

    return _
