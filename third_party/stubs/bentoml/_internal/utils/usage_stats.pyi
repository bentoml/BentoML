from bentoml.utils import ProtoMessageToDict as ProtoMessageToDict
from bentoml.utils.ruamel_yaml import YAML as YAML
from typing import Any

logger: Any

def track(event_type, event_properties: Any | None = ...): ...
def track_save(bento_service, extra_properties: Any | None = ...): ...
