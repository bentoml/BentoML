import socket

from bentoml._internal.utils import create_listen_sock


def test_create_listen_sock():
    host = "127.0.0.1"
    port = 0  # Let OS pick a port
    sock = create_listen_sock(host, port)
    try:
        assert sock.family == socket.AF_INET
        assert sock.type == socket.SOCK_STREAM
        name = sock.getsockname()
        assert name[0] == host
        assert name[1] > 0
    finally:
        sock.close()


def test_create_listen_sock_reuseport():
    host = "127.0.0.1"
    port = 0
    # On Windows, this should still work as it falls back to SO_REUSEADDR
    sock = create_listen_sock(host, port, enable_so_reuseport=True)
    try:
        assert sock.getsockname()[1] > 0
    finally:
        sock.close()
