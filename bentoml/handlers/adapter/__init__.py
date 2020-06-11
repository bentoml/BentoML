from bentoml.handlers.adapter.dataframe_output import DataframeOutput
from bentoml.handlers.adapter.base_output import BaseOutputAdapter


class DefaultOutput(BaseOutputAdapter):
    pass


__all__ = ['DefaultOutput', 'DataframeOutput']
