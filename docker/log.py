import typing as t
import logging

from absl import logging as absl_logging


class ColoredFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    blue: str = "\x1b[34m"
    lightblue: str = "\x1b[36m"
    yellow: str = "\x1b[33m"
    red: str = "\x1b[31m"
    reset: str = "\x1b[0m"
    _format: str = "[%(levelname)s::L%(lineno)d] %(message)s"

    FORMATS = {
        logging.INFO: blue + _format + reset,
        logging.DEBUG: lightblue + _format + reset,
        logging.WARNING: yellow + _format + reset,
        logging.ERROR: red + _format + reset,
        logging.CRITICAL: red + _format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# setup some default
handler = t.cast(logging.Handler, absl_logging.get_absl_handler())
handler.setFormatter(ColoredFormatter())
logger = logging.getLogger(__name__)

# create file handler which logs even debug messages
fh = logging.FileHandler("debug.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
fh.setFormatter(ColoredFormatter())
ch.setFormatter(ColoredFormatter())
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
logger.addHandler(handler)

__all__ = ["logger"]
