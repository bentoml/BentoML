from __future__ import annotations

import abc
import datetime
import enum
import textwrap
import typing as t
import uuid

import attrs
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
    CANCELLED = "cancelled"


@attrs.frozen
class ResultRow(t.Generic[Ti, To]):
    task_id: str
    name: str
    input: Ti
    status: ResultStatus
    result: To


class ResultStore(abc.ABC, t.Generic[Ti, To]):
    @abc.abstractmethod
    async def get_status(self, task_id: str) -> ResultStatus:
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, task_id: str) -> ResultRow[Ti, To]:
        raise NotImplementedError

    async def get_or_none(self, task_id: str) -> t.Optional[ResultRow[Ti, To]]:
        try:
            return await self.get(task_id)
        except (KeyError, RuntimeError):
            return None

    @abc.abstractmethod
    async def new_entry(self, name: str, input: Ti) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def set_result(self, task_id: str, result: To, status: ResultStatus) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def set_status(self, task_id: str, status: ResultStatus) -> None:
        raise NotImplementedError


class Sqlite3Store(ResultStore[Request, Response]):
    RESULT_RETENTION = 60 * 60 * 24  # retain results for 24 hours

    def __init__(self, db_file: str, serializer: Serde | None = None) -> None:
        self._conn = self._connect(db_file)
        if serializer is None:
            serializer = JSONSerde()
        self.serializer = serializer

    def _connect(self, db_file: str) -> Connection:
        import aiosqlite

        return aiosqlite.connect(db_file)

    async def __aenter__(self) -> "t.Self":
        self._conn = await self._conn
        await self._init_db()
        return self

    async def __aexit__(self, *_: t.Any) -> None:
        await self._conn.close()

    async def _init_db(self) -> None:
        await self._conn.execute(
            textwrap.dedent("""\
            CREATE TABLE IF NOT EXISTS result (
                task_id TEXT PRIMARY KEY,
                name TEXT,
                input BLOB,
                status TEXT,
                result BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
        )
        await self._conn.commit()

    async def new_entry(self, name: str, input: Request) -> str:
        task_id = uuid.uuid4().hex
        await self._conn.execute(
            "INSERT INTO result (task_id, name, input, status) VALUES (?, ?, ?, ?)",
            (
                task_id,
                name,
                await self.serializer.serialize_request(input),
                ResultStatus.IN_PROGRESS.value,
            ),
        )
        await self._conn.commit()
        return task_id

    async def get_status(self, task_id: str) -> ResultStatus:
        result = await self._conn.execute(
            "SELECT status FROM result WHERE task_id = ?", (task_id,)
        )
        row = await result.fetchone()
        if row is None:
            raise KeyError(task_id)
        return ResultStatus(row[0])

    async def get(self, task_id: str) -> ResultRow[Request, Response]:
        result = await self._conn.execute(
            "SELECT name, input, status, result FROM result WHERE task_id = ?",
            (task_id,),
        )
        row = await result.fetchone()
        if row is None:
            raise KeyError(task_id)
        if row[2] == ResultStatus.IN_PROGRESS.value:
            raise RuntimeError(f"Task {task_id} is still in progress")
        return ResultRow(
            task_id,
            row[0],
            await self.serializer.deserialize_request(row[1]),
            ResultStatus(row[2]),
            await self.serializer.deserialize_response(row[3]),
        )

    async def set_status(self, task_id: str, status: ResultStatus) -> None:
        await self._conn.execute(
            "UPDATE result SET status = ?, updated_at = ? WHERE task_id = ? AND status = ?",
            (
                status.value,
                datetime.datetime.now(tz=datetime.timezone.utc),
                task_id,
                ResultStatus.IN_PROGRESS.value,
            ),
        )
        await self._conn.commit()

    async def set_result(
        self, task_id: str, result: Response, status: ResultStatus
    ) -> None:
        await self._conn.execute(
            "UPDATE result SET status = ?, result = ?, updated_at = ? WHERE task_id = ?",
            (
                status.value,
                await self.serializer.serialize_response(result),
                datetime.datetime.now(tz=datetime.timezone.utc),
                task_id,
            ),
        )
        # delete older results
        await self._conn.execute(
            "DELETE FROM result WHERE updated_at < ?",
            (
                datetime.datetime.now(tz=datetime.timezone.utc)
                - datetime.timedelta(seconds=self.RESULT_RETENTION),
            ),
        )
        await self._conn.commit()
