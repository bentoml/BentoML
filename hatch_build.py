import os
import subprocess
import warnings
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        webui_root = os.path.join(self.root, "webui")
        try:
            subprocess.run(["pnpm", "install"], check=True, cwd=webui_root)
        except FileNotFoundError:
            warnings.warn("pnpm not found, skipping building webui")
            return
        subprocess.run(["pnpm", "build:copy"], check=True, cwd=webui_root)
