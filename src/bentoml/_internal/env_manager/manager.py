from __future__ import annotations

import logging
import os
import tempfile
import typing as t
import weakref

from simple_di import Provide
from simple_di import inject

from ..bento.bento import Bento
from ..bento.bento import BentoInfo
from ..configuration.containers import BentoMLContainer
from ..utils.filesystem import safe_remove_dir
from .envs import Conda

if t.TYPE_CHECKING:
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
        env_store: str = Provide[BentoMLContainer.env_store_dir],
    ):
        if is_ephemeral:
            if env_name is None:
                env_name = "ephemeral_env"
            env_fs = tempfile.mkdtemp()
            weakref.finalize(self, safe_remove_dir, env_fs)
        else:
            assert env_name is not None, "persistent environments need a valid name."
            env_fs = os.path.join(env_store, env_type)

        if env_type == "conda":
            self.environment = Conda(name=env_name, path=env_fs, bento=bento)
        else:
            raise NotImplementedError(f"'{env_type}' is not supported.")

    @classmethod
    def from_bento(
        cls,
        env_type: t.Literal["conda"],
        bento: Bento,
        is_ephemeral: bool = True,
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
