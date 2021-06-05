import json
from typing import Sequence

from bentoml.adapters.json_output import JsonOutput
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


class DataframeOutput(JsonOutput):
    """
    Output adapters converts returns of user defined API function into specific output,
    such as HTTP response, command line stdout or AWS Lambda event object.

    Args:
        cors (str): DEPRECATED. Moved to the configuration file.
            The value of the Access-Control-Allow-Origin header set in the
            HTTP/AWS Lambda response object. If set to None, the header will not be set.
            Default is None.
        ensure_ascii(bool): Escape all non-ASCII characters. Default False.
        output_orient(str): The orient format of output dataframes.
            Same as pd.DataFrame.to_json(orient=orient)
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
