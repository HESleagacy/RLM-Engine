"""Keep the default suite offline: accidental provider calls must fail locally."""

import socket

import pytest


@pytest.fixture(autouse=True)
def block_network_requests(monkeypatch):
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Network access is disabled in offline tests")
        return original_connect(sock, address)

    def connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise AssertionError("Network access is disabled in offline tests")
        return original_connect_ex(sock, address)

    # Unix-domain IPC remains available to multiprocessing.
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
