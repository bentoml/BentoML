from ..bundle.config import SavedBundleConfig as SavedBundleConfig
from ..bundle.templates import BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE as BENTO_SERVICE_BUNDLE_SETUP_PY_TEMPLATE, INIT_PY_TEMPLATE as INIT_PY_TEMPLATE, MANIFEST_IN_TEMPLATE as MANIFEST_IN_TEMPLATE, MODEL_SERVER_DOCKERFILE_CPU as MODEL_SERVER_DOCKERFILE_CPU
from ..env.local_py_modules import copy_local_py_modules as copy_local_py_modules, copy_zip_import_archives as copy_zip_import_archives
from ..env.pip_pkg import ZIPIMPORT_DIR as ZIPIMPORT_DIR, get_zipmodules as get_zipmodules
from ..exceptions import BentoMLException as BentoMLException
from ..service import BentoService as BentoService
from ..utils import archive_directory_to_tar as archive_directory_to_tar, is_gcs_url as is_gcs_url, is_s3_url as is_s3_url
from ..utils.open_api import get_open_api_spec_json as get_open_api_spec_json
from ..utils.tempdir import TempDirectory as TempDirectory
from ..utils.usage_stats import track_save as track_save
from typing import Any

def versioneer(): ...
def validate_version_str(version_str) -> None: ...

DEFAULT_SAVED_BUNDLE_README: str
logger: Any

def save_to_dir(bento_service: BentoService, path: str, version: str = ..., silent: bool = ...) -> None: ...
def normalize_gztarball(file_path: str): ...
