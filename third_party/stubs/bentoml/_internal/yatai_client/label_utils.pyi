from bentoml.exceptions import BentoMLException as BentoMLException
from bentoml.yatai.proto.label_selectors_pb2 import LabelSelectors as LabelSelectors
from typing import Any

label_expression_operators: Any

def value_string_to_list(value_string): ...
def generate_gprc_labels_selector(label_selectors, label_query) -> None: ...
