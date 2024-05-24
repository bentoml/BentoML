from __future__ import annotations

import abc
import enum
import textwrap
import typing as t
import uuid

from starlette.requests import Request
from starlette.responses import Response

from .serde import JSONSerde
from .serde import Serde

Ti = t.TypeVar("Ti")
To = t.TypeVar("To")

if t.TYPE_CHECKING:
    from aiosqlite import Connection


class ResultStatus(enum.Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"


class ResultStore(abc.ABC, t.Generic[Ti, To]):
    @abc.abstractmethod
    async def get_status(self, task_id: str) -> ResultStatus:
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, task_id: str) -> tuple[To, ResultStatus]:
        raise NotImplementedError

    async def get_or_none(self, task_id: str) -> t.Optional[tuple[To, ResultStatus]]:
        try:
            return await self.get(task_id)
        except (KeyError, RuntimeError):
            return None

    @abc.abstractmethod
    async def new_entry(self, input: Ti) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def set_result(self, task_id: str, result: To, status: ResultStatus) -> None:
        raise NotImplementedError


class Sqlite3Store(ResultStore[Request, Response]):
    def __init__(self, db_file: str, serializer: Serde | None = None) -> None:
        self.db_url = db_file
        self._conn: Connection | None = None
        if serializer is None:
            serializer = JSONSerde()
        self.serializer = serializer

    @property
    def conn(self) -> Connection:
        if self._conn is None:
            raise RuntimeError("Connection not established")
        return self._conn

    async def _connect(self) -> None:
        if self._conn is None:
            import aiosqlite

            self._conn = await aiosqlite.connect(self.db_url)

    async def _disconnect(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "t.Self":
        await self._connect()
        await self._init_db()
        return self

    async def __aexit__(self, *_: t.Any) -> None:
        await self._disconnect()

    async def _init_db(self) -> None:
        await self.conn.execute(
            textwrap.dedent("""\
            CREATE TABLE IF NOT EXISTS result (
                task_id TEXT PRIMARY KEY,
                input BLOB,
                status TEXT,
                result BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        )

    async def new_entry(self, input: Request) -> str:
        task_id = uuid.uuid4().hex
        await self.conn.execute(
            "INSERT INTO result (task_id, input, status) VALUES (?, ?, ?)",
            (
                task_id,
                await self.serializer.serialize_request(input),
                ResultStatus.IN_PROGRESS.value,
            ),
        )
        return task_id

    async def get_status(self, task_id: str) -> ResultStatus:
        result = await self.conn.execute(
            "SELECT status FROM result WHERE task_id = ?", (task_id,)
        )
        row = await result.fetchone()
        if row is None:
            raise KeyError(task_id)
        return ResultStatus(row[0])

    async def get(self, task_id: str) -> tuple[Response, ResultStatus]:
        result = await self.conn.execute(
            "SELECT result, status FROM result WHERE task_id = ?", (task_id,)
        )
        row = await result.fetchone()
        if row is None:
            raise KeyError(task_id)
        if row[1] == ResultStatus.IN_PROGRESS.value:
            raise RuntimeError(f"Task {task_id} is still in progress")
        return await self.serializer.deserialize_response(row[0]), ResultStatus(row[1])

    async def set_result(
        self, task_id: str, result: Response, status: ResultStatus
    ) -> None:
        await self.conn.execute(
            "UPDATE result SET status = ?, result = ? WHERE task_id = ?",
            (
                status,
                await self.serializer.serialize_response(result),
                task_id,
            ),
        )
