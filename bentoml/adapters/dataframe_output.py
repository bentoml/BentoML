import json
from typing import Sequence

from bentoml.adapters.json_output import JsonSerializableOutput
from bentoml.types import InferenceError, InferenceResult, InferenceTask
from bentoml.utils.dataframe_util import PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS


def df_to_json(result, pandas_dataframe_orient="records"):
    import pandas as pd

    assert (
        pandas_dataframe_orient in PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS
    ), f"unknown pandas dataframe orient '{pandas_dataframe_orient}'"

    if isinstance(result, pd.DataFrame):
        return result.to_json(orient=pandas_dataframe_orient)

    if isinstance(result, pd.Series):
        return pd.DataFrame(result).to_json(orient=pandas_dataframe_orient)
    return json.dumps(result)


class DataframeOutput(JsonSerializableOutput):
    """
    Converts result of user defined API function into specific output.

    Args:
        cors (str): The value of the Access-Control-Allow-Origin header set in the
            AWS Lambda response object. Default is "*". If set to None,
            the header will not be set.
    """

    BATCH_MODE_SUPPORTED = True

    def __init__(self, output_orient='records', **kwargs):
        super().__init__(**kwargs)
        self.output_orient = output_orient

        assert self.output_orient in PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS, (
            f"Invalid 'output_orient'='{self.orient}', valid options are "
            f"{PANDAS_DATAFRAME_TO_JSON_ORIENT_OPTIONS}"
        )

    @property
    def config(self):
        base_config = super(DataframeOutput, self).config
        return dict(base_config, output_orient=self.output_orient,)

    @property
    def pip_dependencies(self):
        """
        :return: List of PyPI package names required by this OutputAdapter
        """
        return ['pandas']

    def pack_user_func_return_value(
        self, return_result, tasks: Sequence[InferenceTask],
    ) -> Sequence[InferenceResult[str]]:
        rv = []
        i = 0
        for task in tasks:
            if task.batch is None:
                result = return_result[i : i + 1]
                i += 1
            else:
                result = return_result[i : i + task.batch]
                i += task.batch
            try:
                result = df_to_json(result, self.output_orient)
                rv.append(InferenceResult(http_status=200, data=result))
            except Exception as e:  # pylint: disable=broad-except
                rv.append(InferenceError(err_msg=str(e), http_status=500))
        return rv
