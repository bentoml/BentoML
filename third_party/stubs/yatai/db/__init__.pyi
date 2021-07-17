from typing import Any
from yatai.yatai.db.base import Base as Base
from yatai.yatai.db.stores.deployment import DeploymentStore as DeploymentStore
from yatai.yatai.db.stores.label import LabelStore as LabelStore
from yatai.yatai.db.stores.metadata import MetadataStore as MetadataStore

logger: Any

def is_postgresql_db(db_url): ...
def is_sqlite_db(db_url): ...

class DB:
    db_url: Any
    engine: Any
    session_maker: Any
    def __init__(self, db_url) -> None: ...
    def create_session(self) -> None: ...
    def create_all_or_upgrade_db(self) -> None: ...
