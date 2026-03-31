from __future__ import annotations

import pytest

from bentoml._internal.container.frontend.dockerfile import (
    ALLOWED_CUDA_VERSION_ARGS,
    SUPPORTED_CUDA_VERSIONS,
    get_cuda_base_image,
)
from bentoml._internal.bento.build_config import DockerOptions


class TestSupportedCudaVersions:
    """Ensure every version in ALLOWED_CUDA_VERSION_ARGS resolves to a
    version in SUPPORTED_CUDA_VERSIONS."""

    def test_allowed_args_resolve_to_supported(self):
        for arg, resolved in ALLOWED_CUDA_VERSION_ARGS.items():
            assert resolved in SUPPORTED_CUDA_VERSIONS, (
                f"ALLOWED_CUDA_VERSION_ARGS['{arg}'] -> '{resolved}' "
                f"is not in SUPPORTED_CUDA_VERSIONS"
            )


class TestGetCudaBaseImage:
    """Verify that get_cuda_base_image produces valid NVIDIA image tags."""

    @pytest.mark.parametrize(
        "cuda_version, expected_image",
        [
            ("12.8.1", "nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04"),
            ("12.8.0", "nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04"),
            ("12.6.3", "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"),
            ("12.6.0", "nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04"),
        ],
    )
    def test_debian_new_cuda_versions_use_ubuntu2404(self, cuda_version, expected_image):
        assert get_cuda_base_image("debian", cuda_version) == expected_image

    @pytest.mark.parametrize(
        "cuda_version, expected_image",
        [
            ("12.1.1", "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"),
            ("12.0.0", "nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu22.04"),
            ("11.8.0", "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04"),
            ("11.6.2", "nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu22.04"),
            ("11.2.2", "nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu22.04"),
        ],
    )
    def test_debian_old_cuda_versions_use_ubuntu2204(self, cuda_version, expected_image):
        assert get_cuda_base_image("debian", cuda_version) == expected_image

    def test_ubi8_uses_static_template(self):
        assert (
            get_cuda_base_image("ubi8", "12.1.1")
            == "nvidia/cuda:12.1.1-cudnn8-runtime-ubi8"
        )

    def test_distro_without_cuda_raises(self):
        with pytest.raises(Exception, match="does not support CUDA"):
            get_cuda_base_image("alpine", "12.8.1")


class TestDockerOptionsCudaVersion:
    """Verify that DockerOptions properly validates and converts cuda_version."""

    @pytest.mark.parametrize("shorthand, expected", [("12", "12.8.1"), ("12.6", "12.6.3"), ("12.8", "12.8.1")])
    def test_shorthand_resolves(self, shorthand, expected):
        opts = DockerOptions(cuda_version=shorthand)
        assert opts.cuda_version == expected

    def test_invalid_version_raises(self):
        with pytest.raises(Exception):
            DockerOptions(cuda_version="99.0.0")

    def test_none_stays_none(self):
        opts = DockerOptions(cuda_version=None)
        assert opts.cuda_version is None
