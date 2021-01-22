import pickle
from functools import lru_cache
from typing import Sequence

from bentoml import config as bentoml_config
from bentoml.types import HTTPRequest, HTTPResponse

BATCH_REQUEST_HEADER = bentoml_config("apiserver").get("batch_request_header")

import io
import builtins

safe_builtins = {
    'range',
    'complex',
    'set',
    'frozenset',
    'slice',
}


class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        """Only allow safe classes from builtins"""
        if module == "builtins" and name in safe_builtins:
            return getattr(builtins, name)
        """Forbid everything else"""
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" %
                                     (module, name))

def restricted_loads(s):
    """Helper function analogous to pickle.loads()"""
    return RestrictedUnpickler(io.BytesIO(s)).load()


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
        restricted_loads(raw)
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
