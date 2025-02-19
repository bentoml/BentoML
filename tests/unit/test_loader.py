from pathlib import Path

import pytest

from _bentoml_impl.loader import import_service
from _bentoml_impl.loader import normalize_identifier
from bentoml.exceptions import ImportServiceError


def test_normalize_identifier_py_file(tmp_path: Path):
    service_file = tmp_path / "my_service.py"
    service_file.write_text("")  # Create empty file

    module_name, path = normalize_identifier(str(service_file))
    assert module_name == "my_service"
    assert path == tmp_path  # Since we return path.parent for .py files


def test_normalize_identifier_service_py_backward_compatibility(tmp_path: Path):
    """Test backward compatibility with service.py in directory."""
    service_dir = tmp_path / "myproject"
    service_dir.mkdir()
    service_file = service_dir / "service.py"
    service_file.write_text("")  # Create empty file

    module_name, path = normalize_identifier(str(service_dir))
    assert module_name == "service"
    assert path == service_dir


def test_import_service_multiple_services_error(tmp_path: Path):
    service_file = tmp_path / "service_with_multiple.py"
    service_file.write_text("""
import bentoml

@bentoml.service
class Service1:
    pass

@bentoml.service
class Service2:
    pass
""")

    with pytest.raises(ImportServiceError) as exc:
        import_service("service_with_multiple", bento_path=tmp_path)

    error_msg = str(exc.value)
    assert "Multiple services found in the module" in error_msg
    assert "Available services:" in error_msg
    assert "SERVICE_NAME" in error_msg
    assert "Service1" in error_msg
    assert "Service2" in error_msg
