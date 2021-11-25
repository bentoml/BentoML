"""
Unopinionated display configuration.
"""
_initial_defencoding: str | None = ...

def detect_console_encoding() -> str:
    """
    Try to find the most capable encoding supported by the console.
    slightly modified from the way IPython handles the same issue.
    """
    ...

pc_encoding_doc = ...
