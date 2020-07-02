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

from typing import Iterable

import argparse

from bentoml.exceptions import BentoMLException
from bentoml.utils.dataframe_util import PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS
from bentoml.marshal.utils import SimpleResponse, SimpleRequest
from bentoml.adapters.base_output import BaseOutputAdapter


def df_to_json(result, pandas_dataframe_orient="records"):
    import pandas as pd

    assert (
        pandas_dataframe_orient in PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS
    ), f"unkown pandas dataframe orient '{pandas_dataframe_orient}'"

    if isinstance(result, pd.DataFrame):
        return result.to_json(orient=pandas_dataframe_orient)

    if isinstance(result, pd.Series):
        return pd.DataFrame(result).to_json(orient=pandas_dataframe_orient)
    raise BentoMLException("DataframeOutput only accepts pd.Series or pd.DataFrame.")


class DataframeOutput(BaseOutputAdapter):
    """
    Converts result of use defined API function into specific output.

    Args:
        output_orient (str): Prefer json orient format for output result. Default is
            records.
        cors (str): The value of the Access-Control-Allow-Origin header set in the
            AWS Lambda response object. Default is "*". If set to None,
            the header will not be set.
    """

    def __init__(self, output_orient='records', **kwargs):
        super(DataframeOutput, self).__init__(**kwargs)
        self.output_orient = output_orient

        assert self.output_orient in PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS, (
            f"Invalid 'output_orient'='{self.orient}', valid options are "
            f"{PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS}"
        )

    @property
    def config(self):
        base_config = super(DataframeOutput, self).config
        return dict(base_config, output_orient=self.output_orient,)

    def to_batch_response(
        self,
        result_conc,
        slices=None,
        fallbacks=None,
        requests: Iterable[SimpleRequest] = None,
    ) -> Iterable[SimpleResponse]:
        # TODO(bojiang): header content_type

        if slices is None:
            slices = [i for i, _ in enumerate(result_conc)]
        if fallbacks is None:
            fallbacks = [None] * len(slices)
        responses = [None] * len(slices)

        for i, (s, f) in enumerate(zip(slices, fallbacks)):
            if s is None:
                responses[i] = f
                continue
            result = result_conc[s]
            try:
                json_output = df_to_json(
                    result, pandas_dataframe_orient=self.output_orient
                )
                responses[i] = SimpleResponse(
                    200, (("Content-Type", "application/json"),), json_output
                )
            except AssertionError as e:
                responses[i] = SimpleResponse(400, None, str(e))
            except Exception as e:  # pylint: disable=broad-except
                responses[i] = SimpleResponse(500, None, str(e))
        return responses

    def to_cli(self, result, args):
        """Handles an CLI command call, convert CLI arguments into
        corresponding data format that user API function is expecting, and
        prints the API function result to console output

        :param args: CLI arguments
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-o", "--output", default="str", choices=["str", "json"])
        parser.add_argument(
            "--output_orient",
            default=self.output_orient,
            choices=PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS,
        )
        parsed_args = parser.parse_args(args)

        if parsed_args.output == 'json':
            result = df_to_json(
                result, pandas_dataframe_orient=parsed_args.output_orient
            )
        else:
            result = str(result)
        print(result)

    def to_aws_lambda_event(self, result, event):

        result = df_to_json(result, pandas_dataframe_orient=self.output_orient)

        # Allow disabling CORS by setting it to None
        if self.cors:
            return {
                "statusCode": 200,
                "body": result,
                "headers": {"Access-Control-Allow-Origin": self.cors},
            }

        return {"statusCode": 200, "body": result}

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this InputAdapter
        """
        return ['pandas']
