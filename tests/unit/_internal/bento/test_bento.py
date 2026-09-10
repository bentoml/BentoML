# pylint: disable=unused-argument
from __future__ import annotations

import os
import posixpath
from datetime import datetime
from datetime import timezone
from sys import version_info
from typing import TYPE_CHECKING

import pytest

from bentoml import Tag
from bentoml import bentos
from bentoml._internal.bento import Bento
from bentoml._internal.bento.bento import BaseBentoInfo
from bentoml._internal.bento.bento import BentoApiInfo
from bentoml._internal.bento.bento import BentoInfo
from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.bento.bento import BentoRunnerInfo
from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.configuration import BENTOML_VERSION
from bentoml._internal.models import ModelStore

if TYPE_CHECKING:
    from pathlib import Path


def test_bento_info(tmpdir: Path):
    start = datetime.now(timezone.utc)
    bentoinfo_a = BentoInfo(tag=Tag("tag"), service="service")
    end = datetime.now(timezone.utc)

    assert bentoinfo_a.bentoml_version == BENTOML_VERSION
    assert start <= bentoinfo_a.creation_time <= end
    # validate should fail

    tag = Tag("test", "version")
    service = "testservice"
    labels = {"label": "stringvalue"}
    model_creation_time = datetime.now(timezone.utc)
    model_a = BentoModelInfo(
        tag=Tag("model_a", "v1"),
        module="model_a_module",
        creation_time=model_creation_time,
    )
    model_b = BentoModelInfo(
        tag=Tag("model_b", "v3"),
        module="model_b_module",
        creation_time=model_creation_time,
        alias="model_b_alias",
    )
    models = [model_a, model_b]
    runner_a = BentoRunnerInfo(
        name="runner_a",
        runnable_type="test_runnable_a",
        models=["runner_a_model"],
        resource_config={"cpu": 2},
    )
    runners = [runner_a]
    api_predict = BentoApiInfo(
        name="predict",
        input_type="NumpyNdarray",
        output_type="NumpyNdarray",
    )
    apis = [api_predict]

    bentoinfo_b = BentoInfo(
        tag=tag,
        service=service,
        labels=labels,
        runners=runners,
        models=models,
        apis=apis,
    )

    bento_yaml_b_filename = os.path.join(tmpdir, "b_dump.yml")
    with open(bento_yaml_b_filename, "w", encoding="utf-8") as bento_yaml_b:
        bentoinfo_b.dump(bento_yaml_b)

    expected_yaml = """\
service: testservice
name: test
version: version
bentoml_version: {bentoml_version}
creation_time: '{creation_time}'
labels:
  label: stringvalue
models:
- tag: model_a:v1
  module: model_a_module
  creation_time: '{model_creation_time}'
- tag: model_b:v3
  module: model_b_module
  creation_time: '{model_creation_time}'
  alias: model_b_alias
entry_service: ''
services: []
envs: []
schema: {{}}
args: {{}}
spec: 1
runners:
- name: runner_a
  runnable_type: test_runnable_a
  embedded: false
  models:
  - runner_a_model
  resource_config:
    cpu: 2
apis:
- name: predict
  input_type: NumpyNdarray
  output_type: NumpyNdarray
docker:
  distro: debian
  python_version: '{python_version}'
  cuda_version: null
  env: null
  system_packages: null
  setup_script: null
  base_image: null
  dockerfile_template: null
python:
  requirements_txt: null
  packages: null
  lock_packages: true
  pack_git_packages: true
  index_url: null
  no_index: null
  trusted_host: null
  find_links: null
  extra_index_url: null
  pip_args: null
  wheels: null
  is_src_layout: false
conda:
  environment_yml: null
  channels: null
  dependencies: null
  pip: null
"""

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        assert bento_yaml_b.read() == expected_yaml.format(
            bentoml_version=BENTOML_VERSION,
            creation_time=bentoinfo_b.creation_time.isoformat(),
            model_creation_time=model_creation_time.isoformat(),
            python_version=f"{version_info.major}.{version_info.minor}",
        )

    with open(bento_yaml_b_filename, encoding="utf-8") as bento_yaml_b:
        bentoinfo_b_from_yaml = BaseBentoInfo.from_yaml_file(bento_yaml_b)

        assert bentoinfo_b_from_yaml == bentoinfo_b


def build_test_bento() -> Bento:
    bento_cfg = BentoBuildConfig(
        "simplebento.py:SimpleBento",
        include=["*.py", "config.json", "somefile", "*dir*", ".bentoignore"],
        exclude=["*.storage", "/somefile", "/subdir2"],
        conda={
            "environment_yml": "./environment.yaml",
        },
        docker={
            "setup_script": "./setup_docker_container.sh",
        },
        labels={
            "team": "foo",
            "dataset_version": "abc",
            "framework": "pytorch",
        },
        models=["testmodel"],
    )

    return Bento.create(bento_cfg, version="1.0", build_ctx="./simplebento")


@pytest.mark.usefixtures("change_test_dir")
def test_bento_export(tmp_path: Path, model_store: ModelStore):
    working_dir = os.getcwd()

    testbento = build_test_bento()
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    cfg = BentoBuildConfig("bentoa.py:BentoA")
    bentoa = Bento.create(cfg, build_ctx="./bentoa")
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    bentoa1 = Bento.create(cfg, build_ctx="./bentoa1")
    # Bento build will change working dir to the build_context, this will reset it
    os.chdir(working_dir)

    cfg = BentoBuildConfig("bentob.py:BentoB")
    bentob = Bento.create(cfg, build_ctx="./bentob")

    bento = testbento
    path = posixpath.join(tmp_path, "testbento")
    export_path = bento.export(path)
    assert export_path == path.replace(os.sep, "/") + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = bentoa
    path = posixpath.join(tmp_path, "bentoa")
    export_path = bento.export(path)
    assert export_path == path.replace(os.sep, "/") + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = bentoa1
    path = posixpath.join(tmp_path, "bentoa1")
    export_path = bento.export(path)
    assert export_path == path.replace(os.sep, "/") + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = bentob
    path = posixpath.join(tmp_path, "bentob")
    export_path = bento.export(path)
    assert export_path == path.replace(os.sep, "/") + ".bento"
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    bento = testbento
    path = posixpath.join(tmp_path, "testbento.bento")
    export_path = bento.export(path)
    assert export_path == path.replace(os.sep, "/")
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = posixpath.join(tmp_path, "testbento-parent")
    os.mkdir(path)
    export_path = bento.export(path)
    assert export_path == posixpath.join(path, bento._export_name + ".bento").replace(
        os.sep, "/"
    )
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = posixpath.join(tmp_path, "testbento-parent-2/")
    with pytest.raises(FileNotFoundError):
        export_path = bento.export(path)

    path = posixpath.join(tmp_path, "bento-dir")
    os.mkdir(path)
    export_path = bento.export(path)
    assert export_path == posixpath.join(path, bento._export_name + ".bento").replace(
        os.sep, "/"
    )
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = "file://" + posixpath.join(tmp_path, "testbento-by-url")
    export_path = bento.export(path)
    assert export_path == posixpath.join(tmp_path, "testbento-by-url.bento").replace(
        os.sep, "/"
    )
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento
    imported_bento = Bento.import_from(path + ".bento")
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = "file://" + posixpath.join(tmp_path, "testbento-by-url")
    with pytest.raises(ValueError):
        bento.export(path, subpath="/badsubpath")

    path = "zip://" + posixpath.join(tmp_path, "testbento.zip")
    export_path = bento.export(path)
    assert export_path == path[6:].replace(os.sep, "/")
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = posixpath.join(tmp_path, "testbento-gz")
    os.mkdir(path)
    export_path = bento.export(path, output_format="gz")
    assert export_path == posixpath.join(path, bento._export_name + ".gz").replace(
        os.sep, "/"
    )
    assert os.path.isfile(export_path)
    imported_bento = Bento.import_from(export_path)
    assert imported_bento.tag == bento.tag
    assert imported_bento.info == bento.info
    del imported_bento

    path = posixpath.join(tmp_path, "testbento-gz-1/")
    with pytest.raises(FileNotFoundError):
        bento.export(path, output_format="gz")


@pytest.mark.usefixtures("change_test_dir")
def test_export_bento_with_models(model_store: ModelStore, tmp_path: Path):
    working_dir = os.getcwd()
    bento = build_test_bento()
    os.chdir(working_dir)

    assert bento._model_store is None
    model_tag = bento.info.models[0].tag
    path = os.path.join(tmp_path, "testbento.bento")
    exported_path = bento.export(path)
    # clear models
    model_store.delete(model_tag)
    imported_bento = Bento.import_from(exported_path).save()
    assert imported_bento._model_store is None
    assert model_store.get(model_tag) is not None
    bentos.delete(imported_bento.tag)


@pytest.mark.usefixtures("change_test_dir")
def test_bento(model_store: ModelStore):
    start = datetime.now(timezone.utc)
    bento = build_test_bento()
    end = datetime.now(timezone.utc)

    assert bento.info.bentoml_version == BENTOML_VERSION
    assert start <= bento.creation_time <= end
    # validate should fail

    def list_bento(path: str) -> set[str]:
        return set(os.listdir(bento.path_of(path)))

    assert list_bento("/") == {
        "bento.yaml",
        "apis",
        "README.md",
        "src",
        "env",
    }
    assert list_bento("src") == {
        "simplebento.py",
        "subdir",
        "bentofile.yaml",
        ".bentoignore",
    }
    assert list_bento("src/subdir") == {"somefile"}


@pytest.mark.usefixtures("change_test_dir")
def test_build_bento_with_args():
    from bentoml._internal.configuration.containers import BentoMLContainer

    bento = bentos.build_bentofile(
        build_ctx="./bento_with_args", args={"label": "awesome"}
    )
    BentoMLContainer.bento_arguments.reset()
    assert bento.info.args == {"label": "awesome"}


def test_uv_workspace_support(tmp_path: Path):
    from bentoml._internal.bento.build_config import PythonOptions
    from bentoml._internal.bento.build_config import find_workspace_root
    from bentoml._internal.bento.build_config import get_workspace_members
    from bentoml._internal.bento.build_config import rewrite_if_workspace_member

    # 1. Setup workspace structure
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    # root pyproject.toml
    root_pyproject = workspace_root / "pyproject.toml"
    root_pyproject.write_text(
        "[tool.uv.workspace]\n"
        'members = ["packages/*", "my-app"]\n'
        'exclude = ["**/excluded-member"]\n'
    )

    # packages
    packages_dir = workspace_root / "packages"
    packages_dir.mkdir()

    bird_feeder = packages_dir / "bird-feeder"
    bird_feeder.mkdir()
    (bird_feeder / "pyproject.toml").write_text(
        "[project]\n"
        'name = "bird-feeder"\n'
        'version = "0.1.0"\n'
        'dependencies = ["worm-catcher"]\n'
        "[tool.uv.sources]\n"
        "worm-catcher = { workspace = true }\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
    )
    (bird_feeder / "src").mkdir()
    (bird_feeder / "src" / "__init__.py").touch()

    worm_catcher = packages_dir / "worm-catcher"
    worm_catcher.mkdir()
    (worm_catcher / "pyproject.toml").write_text(
        "[project]\n"
        'name = "worm-catcher"\n'
        'version = "0.1.0"\n'
        "dependencies = []\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
    )
    (worm_catcher / "src").mkdir()
    (worm_catcher / "src" / "__init__.py").touch()

    excluded_member = packages_dir / "excluded-member"
    excluded_member.mkdir()
    (excluded_member / "pyproject.toml").write_text(
        '[project]\nname = "excluded-member"\nversion = "0.1.0"\n'
    )

    my_app = workspace_root / "my-app"
    my_app.mkdir()
    (my_app / "pyproject.toml").write_text(
        "[project]\n"
        'name = "my-app"\n'
        'version = "0.1.0"\n'
        'dependencies = ["bird-feeder"]\n'
        "[tool.uv.sources]\n"
        "bird-feeder = { workspace = true }\n"
        "[build-system]\n"
        'requires = ["hatchling"]\n'
        'build-backend = "hatchling.build"\n'
    )
    (my_app / "service.py").write_text("import bentoml\nsvc = bentoml.Service('svc')\n")

    # 2. Test helpers
    assert find_workspace_root(str(my_app)) == str(workspace_root)
    assert find_workspace_root(str(bird_feeder)) == str(workspace_root)

    members = get_workspace_members(str(workspace_root))
    assert "my-app" in members
    assert "bird-feeder" in members
    assert "worm-catcher" in members
    assert "excluded-member" not in members

    # Test rewrite helper
    members_map = {
        "bird-feeder": str(bird_feeder),
        "worm-catcher": str(worm_catcher),
    }
    # Package without version or extra
    assert rewrite_if_workspace_member("bird-feeder", members_map) == str(bird_feeder)
    # Package with extras
    assert (
        rewrite_if_workspace_member("bird-feeder[dev]", members_map)
        == f"{bird_feeder!s}[dev]"
    )
    # Package with environment marker
    assert (
        rewrite_if_workspace_member(
            'bird-feeder ; python_version >= "3.10"', members_map
        )
        == f'{bird_feeder!s} ; python_version >= "3.10"'
    )
    # Package with extra and marker
    assert (
        rewrite_if_workspace_member(
            'bird-feeder[dev] ; python_version >= "3.10"', members_map
        )
        == f'{bird_feeder!s}[dev] ; python_version >= "3.10"'
    )
    # Non-member package
    assert rewrite_if_workspace_member("tqdm>=4", members_map) == "tqdm>=4"

    # 3. Test Security Boundary Check in fix_dep_urls
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "pyproject.toml").write_text(
        "[project]\nname = 'outside-pkg'\nversion = '1.0'\n"
    )

    from bentoml.exceptions import BentoMLException

    # We expect BentoMLException to be raised when referencing a directory outside the workspace root
    req_file = tmp_path / "reqs.txt"
    req_file.write_text(f"outside-pkg @ {outside_dir.as_uri()}\n")
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()

    with pytest.raises(BentoMLException) as excinfo:
        PythonOptions.fix_dep_urls(
            str(req_file),
            str(wheels_dir),
            pack_git_packages=False,
            workspace_root=str(workspace_root),
        )
    assert "Security violation" in str(excinfo.value)

    # 4. Zero-pollution Assertion
    def get_file_snapshot(dir_path):
        snapshot = set()
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                rel = os.path.relpath(os.path.join(root, file), dir_path)
                snapshot.add(rel)
        return snapshot

    before_snapshot = get_file_snapshot(str(bird_feeder))

    from bentoml._internal.bento.bentoml_builder import build_local_dep

    out_wheels = tmp_path / "out-wheels"
    out_wheels.mkdir()
    build_local_dep(str(bird_feeder), str(out_wheels))

    after_snapshot = get_file_snapshot(str(bird_feeder))
    assert before_snapshot == after_snapshot, (
        f"Source directory was polluted! Diff: {after_snapshot - before_snapshot}"
    )


def test_uv_workspace_symlink_escape(tmp_path: Path):
    import sys

    from bentoml._internal.bento.build_config import PythonOptions
    from bentoml.exceptions import BentoMLException

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "pyproject.toml").write_text(
        "[project]\nname = 'outside-pkg'\nversion = '1.0'\n"
    )

    symlink_dir = workspace_root / "symlink-dir"
    try:
        symlink_dir.symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        if sys.platform == "win32":
            pytest.skip(
                "Symlink creation not supported on Windows without Developer Mode/Admin rights"
            )
        else:
            raise

    req_file_symlink = tmp_path / "reqs_symlink.txt"
    req_file_symlink.write_text(f"outside-pkg @ {symlink_dir.as_uri()}\n")
    wheels_dir = tmp_path / "wheels"
    wheels_dir.mkdir()

    with pytest.raises(BentoMLException) as excinfo_sym:
        PythonOptions.fix_dep_urls(
            str(req_file_symlink),
            str(wheels_dir),
            pack_git_packages=False,
            workspace_root=str(workspace_root),
        )
    assert "Security violation" in str(excinfo_sym.value)
