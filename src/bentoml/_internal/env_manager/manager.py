from __future__ import annotations

import typing as t
import logging

import fs
from simple_di import inject
from simple_di import Provide

from .envs import Conda
from ..bento.bento import Bento
from ..bento.bento import BentoInfo
from ..configuration.containers import BentoMLContainer

if t.TYPE_CHECKING:
    from fs.base import FS

    from .envs import Environment

logger = logging.getLogger(__name__)


class EnvManager:
    environment: Environment

    @inject
    def __init__(
        self,
        env_type: t.Literal["conda"],
        bento: Bento,
        is_ephemeral: bool = True,
        env_name: str | None = None,
        env_store: FS = Provide[BentoMLContainer.env_store],
    ):
        if not is_ephemeral:
            assert env_name is not None, "persistent environments need a valid name."
        if not env_store.exists(env_type):
            env_store.makedir(env_type)
        env_fs = fs.open_fs("temp://") if is_ephemeral else env_store.opendir(env_type)

        if env_type == "conda":
            self.environment = Conda(name=env_name, env_fs=env_fs, bento=bento)
        else:
            raise NotImplementedError(f"'{env_type}' is not supported.")

    @classmethod
    def from_bento(
        cls,
        env_type: t.Literal["conda"],
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
