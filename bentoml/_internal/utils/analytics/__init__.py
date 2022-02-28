from .usage_stats import track
from .usage_stats import get_serve_info
from .usage_stats import scheduled_track
from .usage_stats import BENTOML_DO_NOT_TRACK

CLI_TRACK_EVENT_TYPE = "bentoml_cli"
MODEL_SAVE_TRACK_EVENT_TYPE = "bentoml_model_save"
BENTO_BUILD_TRACK_EVENT_TYPE = "bentoml_bento_build"
BENTO_SERVE_TRACK_EVENT_TYPE = "bentoml_bento_serve_init"

__all__ = [
    "track",
    "scheduled_track",
    "get_serve_info",
    "BENTOML_DO_NOT_TRACK",
]
