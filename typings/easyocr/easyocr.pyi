import typing as t

from numpy.typing import NDArray

class Reader:
    def __init__(
        self,
        lang_list: list[str],
        gpu: bool | str = ...,
        model_storage_directory: str | None = ...,
        user_network_directory: str | None = ...,
        detect_network: str = ...,
        recog_network: str = ...,
        download_enabled: bool = ...,
        detector: bool = ...,
        recognizer: bool = ...,
        verbose: bool = ...,
        quantize: bool = ...,
        cudnn_benchmark: bool = ...,
    ) -> None:
        """Create an EasyOCR Reader

        Parameters:
            lang_list (list): Language codes (ISO 639) for languages to be recognized during analysis.

            gpu (bool): Enable GPU support (default)

            model_storage_directory (string): Path to directory for model data. If not specified,
            models will be read from a directory as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            user_network_directory (string): Path to directory for custom network architecture.
            If not specified, it is as defined by the environment variable
            EASYOCR_MODULE_PATH (preferred), MODULE_PATH (if defined), or ~/.EasyOCR/.

            download_enabled (bool): Enabled downloading of model data via HTTP (default).
        """
        ...

    def readtext(
        self,
        image: str | NDArray[t.Any] | t.IO[bytes],
        decoder: str = ...,
        beamWidth: int = ...,
        batch_size: int = ...,
        workers: int = ...,
        allowlist: list[str] | None = ...,
        blocklist: list[str] | None = ...,
        detail: int = ...,
        rotation_info: t.Any | None = ...,
        paragraph: bool = ...,
        min_size: int = ...,
        contrast_ths: float = ...,
        adjust_contrast: float = ...,
        filter_ths: float = ...,
        text_threshold: float = ...,
        low_text: float = ...,
        link_threshold: float = ...,
        canvas_size: int = ...,
        mag_ratio: float = ...,
        slope_ths: float = ...,
        ycenter_ths: float = ...,
        height_ths: float = ...,
        width_ths: float = ...,
        y_ths: float = ...,
        x_ths: float = ...,
        add_margin: float = ...,
        threshold: float = ...,
        bbox_min_score: float = ...,
        bbox_min_size: int = ...,
        max_candidates: int = ...,
        output_format: str = ...,
    ) -> list[tuple[list[t.Any], str, float]]: ...
