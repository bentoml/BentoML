#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import os
import pathlib

START_COMMENT=f"# {os.path.basename(__file__)}: start\n"
END_COMMENT=f"# {os.path.basename(__file__)}: end\n"
ROOT=pathlib.Path(__file__).parent.parent
_TARGET_FILE=ROOT/"src"/"bentoml"/"llm.py"

def main() -> int:
    try: importlib.metadata.version("openllm")
    except importlib.metadata.PackageNotFoundError:
        print("Failed to find openllm. Make sure to have openllm available locally.")
        return 1
    import openllm
    with _TARGET_FILE.open("r") as f: processed = f.readlines()
    start, end = processed.index(START_COMMENT), processed.index(END_COMMENT)
    lns: list[str] = ["def __dir__()->list[str]:return sorted(openllm._llm.__all__)\n"]
    lns.extend([f"@overload\ndef __getattr__(item: Literal['openllm.{it}'])->t.Any:...\n" for it in openllm._llm.__all__])
    lns.append("def __getattr__(item: t.Any)->t.Any:return getattr(openllm, item)\n")
    processed = processed[:start] + [START_COMMENT, *lns, END_COMMENT] + processed[end+1:]
    with _TARGET_FILE.open("w") as f: f.writelines(processed)
    return 0

if __name__=="__main__": raise SystemExit(main())
