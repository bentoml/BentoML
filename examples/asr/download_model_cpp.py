#!/usr/bin/env python
import os

prev_value = None
if "XDG_DATA_HOME" in os.environ:
    prev_value = os.environ.pop("XDG_DATA_HOME")

if os.environ.get("BENTO_PATH"):
    os.environ["XDG_DATA_HOME"] = os.path.expandvars("/home/bentoml/.local/share")
else:
    os.environ["XDG_DATA_HOME"] = os.path.expandvars("$HOME/.local/share")

from whispercpp import Whisper  # noqa

model_name = os.environ.get("GGML_MODEL", "tiny.en")

Whisper.from_pretrained(model_name)

# restore to previous value if it was set
if prev_value is not None:
    os.environ["XDG_DATA_HOME"] = prev_value
