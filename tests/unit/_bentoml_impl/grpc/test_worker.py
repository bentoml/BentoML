from __future__ import annotations

import pytest


def test_worker_sets_arguments_before_loading_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from _bentoml_impl.worker.grpc_service import main
    from bentoml._internal.configuration.containers import BentoMLContainer

    class LoadStopped(Exception):
        pass

    def fake_load(*args: object, **kwargs: object) -> None:
        assert BentoMLContainer.bento_arguments.get() == {"greeting": "hello"}
        raise LoadStopped

    monkeypatch.setattr("_bentoml_impl.loader.load", fake_load)
    monkeypatch.setattr(
        "bentoml._internal.log.configure_server_logging", lambda: None
    )
    try:
        with pytest.raises(LoadStopped):
            main.main(
                ["service.py:MyService", "--args", '{"greeting": "hello"}'],
                standalone_mode=False,
            )
    finally:
        BentoMLContainer.bento_arguments.reset()
