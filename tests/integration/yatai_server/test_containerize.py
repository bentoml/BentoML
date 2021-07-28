import logging
import subprocess

from bentoml.yatai.client import get_yatai_client

logger = logging.getLogger("bentoml.test")


def test_yatai_server_containerize_without_push(example_bento_service_class):
    svc = example_bento_service_class()
    svc.save()

    yc = get_yatai_client()
    image_tag = "mytag"
    built_tag = yc.repository.containerize(bento=svc.tag, tag=image_tag)
    assert built_tag == f"{image_tag}:{svc.version}"


def test_yatai_server_containerize_from_cli(example_bento_service_class):
    svc = example_bento_service_class()
    svc.save()
    image_tag = "mytagfoo"

    command = [
        "bentoml",
        "containerize",
        svc.tag,
        "--build-arg",
        "EXTRA_PIP_INSTALL_ARGS=--extra-index-url=https://pypi.org",
        "-t",
        image_tag,
    ]
    docker_proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout = docker_proc.stdout.read().decode("utf-8")
    assert f"{image_tag}:{svc.version}" in stdout, "Failed to build container"
