import json


class NumpyJsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, o):  # pylint: disable=method-hidden
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()

        if isinstance(o, np.ndarray):
            return o.tolist()

        return json.JSONEncoder.default(self, o)


PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS = [
    'dict',
    'list',
    'series',
    'split',
    'records',
    'index',
]


def api_func_result_to_json(result, pandas_dataframe_orient="records"):
    try:
        import pandas as pd
    except ImportError:
        pd = None

    assert (
        pandas_dataframe_orient in PANDAS_DATAFRAME_TO_DICT_ORIENT_OPTIONS
    ), f"unkown pandas dataframe orient '{pandas_dataframe_orient}'"

    if pd and isinstance(result, pd.DataFrame):
        return result.to_json(orient=pandas_dataframe_orient)

    if pd and isinstance(result, pd.Series):
        return pd.DataFrame(result).to_dict(orient=pandas_dataframe_orient)

    try:
        return json.dumps(result, cls=NumpyJsonEncoder)
    except (TypeError, OverflowError):
        # when result is not JSON serializable
        return json.dumps({"result": str(result)})
