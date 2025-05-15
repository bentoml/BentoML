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
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@attrs.frozen
class ResultRow:
    task_id: str
    name: str
    status: ResultStatus
    created_at: datetime.datetime
    executed_at: datetime.datetime | None

    def to_json(self) -> dict[str, t.Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
        }


@attrs.frozen
class CompletedResultRow(ResultRow, t.Generic[Ti, To]):
    input: Ti
    result: To
    completed_at: datetime.datetime | None

    def to_json(self) -> dict[str, t.Any]:
        return {
            **super().to_json(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


class ResultStore(abc.ABC, t.Generic[Ti, To]):
    @abc.abstractmethod
    async def get_status(self, task_id: str) -> ResultRow:
        raise NotImplementedError

    @abc.abstractmethod
    async def get(self, task_id: str) -> CompletedResultRow[Ti, To]:
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
        import sqlite3

        import aiosqlite

        return aiosqlite.connect(
            db_file, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )

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
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP DEFAULT NULL
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

    async def get(self, task_id: str) -> CompletedResultRow[Request, Response]:
        result = await self._conn.execute(
            "SELECT name, input, status, result, created_at, executed_at, completed_at "
            "FROM result WHERE task_id = ?",
            (task_id,),
        )
        row = await result.fetchone()
        if row is None:
            raise KeyError(task_id)
        if row[6] is None:
            raise RuntimeError(f"Task {task_id} is not completed")
        return CompletedResultRow(
            task_id,
            row[0],
            ResultStatus(row[2]),
            row[4],
            row[5],
            await self.serializer.deserialize_request(row[1]),
            await self.serializer.deserialize_response(row[3]),
            row[6],
        )

    async def get_status(self, task_id: str) -> ResultRow:
        result = await self._conn.execute(
            "SELECT name, status, created_at, executed_at "
            "FROM result WHERE task_id = ?",
            (task_id,),
        )
        row = await result.fetchone()
        if row is None:
            raise KeyError(task_id)

        return ResultRow(
            task_id,
            row[0],
            ResultStatus(row[1]),
            row[2],
            row[3],
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
            "UPDATE result SET status = ?, result = ?, completed_at = ? WHERE task_id = ?",
            (
                status.value,
                await self.serializer.serialize_response(result),
                datetime.datetime.now(tz=datetime.timezone.utc),
                task_id,
            ),
        )
        # delete older results
        await self._conn.execute(
            "DELETE FROM result WHERE completed_at IS NOT NULL AND completed_at < ?",
            (
                datetime.datetime.now(tz=datetime.timezone.utc)
                - datetime.timedelta(seconds=self.RESULT_RETENTION),
            ),
        )
        await self._conn.commit()
