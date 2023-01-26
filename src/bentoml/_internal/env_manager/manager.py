from __future__ import annotations

import typing as t
import logging

import fs
import attr

from bentoml._internal.bento.bento import Bento

from .envs import Conda
from .envs import Environment
from ..bento.bento import Bento
from ..bento.bento import BentoInfo

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from fs.base import FS


def decode(msg: bytes) -> str:
    if msg:
        return msg.decode("utf-8")
    return ""


@attr.define
class EnvManager:
    env_name: str
    env_type: str
    is_ephemeral: bool
    bento_env_dir: str
    env_fs: FS = attr.field(init=False)
    environment: Environment = attr.field(init=False)

    def __attrs_post_init__(self):
        from bentoml._internal.configuration.containers import BentoMLContainer

        env_home = fs.open_fs(BentoMLContainer.env_store_dir.get())

        if not self.is_ephemeral:
            assert (
                self.env_name is not None
            ), "persistent environments need a valid name."
        if not env_home.exists(self.env_type):
            env_home.makedir(self.env_type)
        self.env_fs = (
            fs.open_fs("temp://")
            if self.is_ephemeral
            else env_home.opendir(self.env_type)
        )

        if self.env_type == "conda":
            self.environment = Conda(
                name=self.env_name, env_fs=self.env_fs, bento_env_dir=self.bento_env_dir
            )

    @classmethod
    def from_bento(
        cls,
        env_type: str,
        bento: Bento,
        is_ephemeral: bool,
    ) -> EnvManager:
        env_name: str
        if is_ephemeral:
            env_name = "ephemeral_env"
        else:
            env_name = str(bento.tag).replace(":", "_")
        return cls(
            env_type=env_type,
            env_name=env_name,
            is_ephemeral=is_ephemeral,
            bento_env_dir=bento.path_of("env"),
        )

    @classmethod
    def from_bentofile(
        cls, env_type: str, bento_info: BentoInfo, is_ephemeral: str
    ) -> EnvManager:
        raise NotImplementedError
