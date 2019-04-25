import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import bentoml  # noqa: E402


def test_custom_api_name():
    # these names should work:
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="a_valid_name")(lambda x: x)
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="AValidName")(lambda x: x)
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="_AValidName")(lambda x: x)
    bentoml.api(bentoml.handlers.DataframeHandler, api_name="a_valid_name_123")(lambda x: x)

    with pytest.raises(ValueError) as e:
        bentoml.api(bentoml.handlers.DataframeHandler, api_name="a invalid name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(ValueError) as e:
        bentoml.api(bentoml.handlers.DataframeHandler, api_name="123_a_invalid_name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")

    with pytest.raises(ValueError) as e:
        bentoml.api(bentoml.handlers.DataframeHandler, api_name="a-invalid-name")(lambda x: x)
    assert str(e.value).startswith("Invalid API name")
