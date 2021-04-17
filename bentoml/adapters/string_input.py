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

import pathlib
from typing import Iterable, Iterator, Sequence, Tuple

import chardet

from bentoml.adapters.base_input import BaseInputAdapter
from bentoml.adapters.utils import decompress_gzip_request
from bentoml.types import AwsLambdaEvent, FileLike, HTTPRequest, InferenceTask

ApiFuncArgs = Tuple[
    Sequence[str],
]


class StringInput(BaseInputAdapter):
    """
    Convert various inputs(HTTP, Aws Lambda or CLI) to strings(list of str), passing it
    to API functions.

    Errors
        400 UnicodeDecodeError

        400 Unsupported charset

    Example Request

    >>> curl -i \\
    >>>    --header "Content-Type: text/plain; charset=utf-8" \\
    >>>    --request POST \\
    >>>    --data 'best movie ever' \\
    >>>    localhost:5000/predict
    """

    BATCH_MODE_SUPPORTED = True

    @decompress_gzip_request
    def from_http_request(self, req: HTTPRequest) -> InferenceTask[str]:
        if req.headers.content_type == 'multipart/form-data':
            _, _, files = HTTPRequest.parse_form_data(req)
            if len(files) != 1:
                return InferenceTask().discard(
                    http_status=400,
                    err_msg=f"BentoML#{self.__class__.__name__} accepts one text file "
                    "at a time",
                )
            input_file = next(iter(files.values()))
            bytes_ = input_file.read()
            charset = chardet.detect(bytes_)['encoding'] or "utf-8"
        else:
            bytes_ = req.body
            charset = req.headers.charset or "utf-8"
        try:
            return InferenceTask(http_headers=req.headers, data=bytes_.decode(charset),)
        except UnicodeDecodeError:
            return InferenceTask().discard(
                http_status=400,
                err_msg=f"{self.__class__.__name__}: UnicodeDecodeError for {req.body}",
            )
        except LookupError:
            return InferenceTask().discard(
                http_status=400,
                err_msg=f"{self.__class__.__name__}: Unsupported charset {req.charset}",
            )

    def from_aws_lambda_event(self, event: AwsLambdaEvent) -> InferenceTask[str]:
        return InferenceTask(aws_lambda_event=event, data=event.get('body', ""),)

    def from_cli(self, cli_args: Tuple[str]) -> Iterator[InferenceTask[str]]:
        import argparse

        parser = argparse.ArgumentParser()
        input_g = parser.add_mutually_exclusive_group(required=True)
        input_g.add_argument('--input', nargs="+", type=str)
        input_g.add_argument('--input-file', nargs="+")

        parsed_args, _ = parser.parse_known_args(list(cli_args))

        for t in self.from_inference_job(
            input_=parsed_args.input, input_file=parsed_args.input_file,
        ):
            t.cli_args = cli_args
            yield t

    def from_inference_job(  # pylint: disable=arguments-differ
        self, input_=None, input_file=None, **extra_args,
    ) -> Iterator[InferenceTask[str]]:
        '''
        Generate InferenceTask from calling bentom_svc.run(input_=None, input_file=None)

        Parameters
        ----------
        input_ : str
            The input value

        input_file : str
            The URI/path of the input file

        extra_args : dict
            Additional parameters
        '''
        if input_file is not None:
            for d in input_file:
                uri = pathlib.Path(d).absolute().as_uri()
                bytes_ = FileLike(uri=uri).read()
                try:
                    charset = chardet.detect(bytes_)['encoding'] or "utf-8"
                    yield InferenceTask(
                        inference_job_args=extra_args, data=bytes_.decode(charset),
                    )
                except UnicodeDecodeError:
                    yield InferenceTask().discard(
                        http_status=400,
                        err_msg=f"{self.__class__.__name__}: "
                        f"Try decoding with {charset} but failed with DecodeError.",
                    )
                except LookupError:
                    yield InferenceTask().discard(
                        http_status=400,
                        err_msg=f"{self.__class__.__name__}: "
                        f"Unsupported charset {charset}",
                    )
        else:
            for d in input_:
                yield InferenceTask(inference_job_args=extra_args, data=d)

    def extract_user_func_args(
        self, tasks: Iterable[InferenceTask[str]]
    ) -> ApiFuncArgs:
        return ([task.data for task in tasks],)
