import pickle
from functools import lru_cache
from typing import Sequence

from bentoml.types import HTTPRequest, HTTPResponse

MARSHAL_REQUEST_HEADER = "BentoML-Is-Merged-Request"


class PlasmaDataLoader:
    """
    Transfer datas with plasma store, in development now
    """

    @classmethod
    @lru_cache(maxsize=1)
    def get_plasma(cls):
        import pyarrow.plasma as plasma
        import subprocess

        object_store_size = 2 * 10 * 1000 * 1000
        plasma_path = "/tmp/store"
        subprocess.Popen(
            ["plasma_store", "-s", plasma_path, "-m", str(object_store_size)]
        )
        return plasma.connect(plasma_path)

    @classmethod
    def merge_requests(cls, reqs) -> bytes:
        merged_reqs = tuple((b, h) for h, b in reqs)
        oid = cls.get_plasma().put(merged_reqs)
        return oid.binary()

    @classmethod
    def split_responses(cls, raw: bytes):
        import pyarrow.plasma as plasma

        oid = plasma.ObjectID(raw)
        merged_responses = cls.get_plasma().get(oid)
        cls.get_plasma().delete((oid,))
        return merged_responses

    @classmethod
    def split_requests(cls, raw: bytes):
        import pyarrow.plasma as plasma

        oid = plasma.ObjectID(raw)
        info_list = cls.get_plasma().get(oid)
        cls.get_plasma().delete((oid,))
        return info_list

    @classmethod
    def merge_responses(cls, resps) -> bytes:
        merged_resps = tuple((r, tuple()) for r in resps)
        oid = cls.get_plasma().put(merged_resps)
        return oid.binary()


class PickleDataLoader:
    @classmethod
    def merge_requests(cls, reqs: Sequence[HTTPRequest]) -> bytes:
        return pickle.dumps(reqs)

    @classmethod
    def split_requests(cls, raw: bytes) -> Sequence[HTTPRequest]:
        return pickle.loads(raw)

    @classmethod
    def merge_responses(cls, resps: Sequence[HTTPResponse]) -> bytes:
        return pickle.dumps(list(resps))

    @classmethod
    def split_responses(cls, raw: bytes) -> Sequence[HTTPResponse]:
        try:
            return pickle.loads(raw)
        except pickle.UnpicklingError:
            raise ValueError(
                f"Batching result unpacking error: \n {raw[:1000]}"
            ) from None


DataLoader = PickleDataLoader
