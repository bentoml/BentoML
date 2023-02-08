from __future__ import annotations

import logging

import fs
from simple_di import inject
from simple_di import Provide

from .envs import Conda
from ..bento.bento import Bento
from ..bento.bento import BentoInfo
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)


def decode(msg: bytes) -> str:
    if msg:
        return msg.decode("utf-8")
    return ""


class EnvManager:
    @inject
    def __init__(
        self,
        env_name: str,
        env_type: str,
        is_ephemeral: bool,
        bento: Bento,
        env_store_dir: str = Provide[BentoMLContainer.env_store_dir],
    ):

        env_store = fs.open_fs(env_store_dir)

        if not is_ephemeral:
            assert env_name is not None, "persistent environments need a valid name."
        if not env_store.exists(env_type):
            env_store.makedir(env_type)
        env_fs = fs.open_fs("temp://") if is_ephemeral else env_store.opendir(env_type)

        if env_type == "conda":
            self.environment = Conda(name=env_name, env_fs=env_fs, bento=bento)

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
            bento=bento,
        )

    @classmethod
    def from_bentofile(
        cls, env_type: str, bento_info: BentoInfo, is_ephemeral: str
    ) -> EnvManager:
        raise NotImplementedError
