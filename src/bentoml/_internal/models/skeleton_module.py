import typing as t


def _log_warning_msg():
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(
        f"This function is an empty function called from '{__name__}' and will return None."
    )


def get_runnable(*args: t.Any, **kwargs: t.Any) -> None:
    _log_warning_msg()


def load_model(*args: t.Any, **kwargs: t.Any) -> None:
    _log_warning_msg()
