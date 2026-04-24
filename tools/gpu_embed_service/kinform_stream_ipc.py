#!/usr/bin/env python3
from __future__ import annotations

import json
import select
import socket
import time
from dataclasses import dataclass


_HEADER_LEN_BYTES = 8
_MAX_HEADER_NBYTES = 1024 * 1024  # 1 MiB
_MAX_PAYLOAD_NBYTES = 512 * 1024 * 1024  # 512 MiB


class StreamProtocolError(RuntimeError):
    pass


def _recvn(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("Socket closed while reading stream frame.")
        buf.extend(chunk)
    return bytes(buf)


def send_frame(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header_copy = dict(header)
    header_copy["payload_nbytes"] = int(len(payload))
    header_bytes = json.dumps(header_copy, separators=(",", ":"), sort_keys=False).encode("utf-8")
    sock.sendall(len(header_bytes).to_bytes(_HEADER_LEN_BYTES, byteorder="big", signed=False))
    sock.sendall(header_bytes)
    if payload:
        sock.sendall(payload)


def recv_frame(sock: socket.socket) -> tuple[dict, bytes]:
    header_len_raw = _recvn(sock, _HEADER_LEN_BYTES)
    header_len = int.from_bytes(header_len_raw, byteorder="big", signed=False)
    if header_len <= 0:
        raise StreamProtocolError(f"Invalid frame header size: {header_len}")
    if header_len > _MAX_HEADER_NBYTES:
        raise StreamProtocolError(
            f"Frame header size too large: {header_len} > {_MAX_HEADER_NBYTES}"
        )
    header_bytes = _recvn(sock, header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except Exception as exc:
        raise StreamProtocolError("Could not decode JSON frame header.") from exc
    if not isinstance(header, dict):
        raise StreamProtocolError("Frame header must decode to a JSON object.")
    payload_nbytes = int(header.get("payload_nbytes", 0))
    if payload_nbytes < 0:
        raise StreamProtocolError(f"Invalid payload size: {payload_nbytes}")
    if payload_nbytes > _MAX_PAYLOAD_NBYTES:
        raise StreamProtocolError(
            f"Frame payload too large: {payload_nbytes} > {_MAX_PAYLOAD_NBYTES}"
        )
    payload = _recvn(sock, payload_nbytes) if payload_nbytes else b""
    return header, payload


def connect_unix_socket(
    socket_path: str,
    *,
    max_wait_seconds: float = 45.0,
    retry_interval_seconds: float = 0.1,
) -> socket.socket:
    start = time.monotonic()
    last_exc: Exception | None = None
    while True:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(socket_path)
            return sock
        except Exception as exc:  # pragma: no cover - exercised in integration paths
            last_exc = exc
            sock.close()
            if time.monotonic() - start >= max_wait_seconds:
                raise RuntimeError(
                    f"Timed out connecting to stream socket {socket_path}: {last_exc}"
                ) from last_exc
            time.sleep(retry_interval_seconds)


@dataclass
class StreamClient:
    socket_path: str
    max_wait_seconds: float = 45.0
    retry_interval_seconds: float = 0.1
    _sock: socket.socket | None = None

    def connect(self) -> socket.socket:
        if self._sock is None:
            self._sock = connect_unix_socket(
                self.socket_path,
                max_wait_seconds=self.max_wait_seconds,
                retry_interval_seconds=self.retry_interval_seconds,
            )
        return self._sock

    def send(self, header: dict, payload: bytes = b"") -> None:
        send_frame(self.connect(), header, payload)

    def recv(self, timeout_seconds: float | None = None) -> tuple[dict, bytes]:
        sock = self.connect()
        if timeout_seconds is None:
            return recv_frame(sock)
        timeout = max(0.0, float(timeout_seconds))
        prev_timeout = sock.gettimeout()
        try:
            # Do not use socket timeouts during frame reads. A timeout mid-frame
            # can desynchronize the stream protocol by discarding partial bytes.
            sock.settimeout(None)
            readable, _, _ = select.select([sock], [], [], timeout)
            if not readable:
                raise TimeoutError("Timed out waiting for stream frame.")
            return recv_frame(sock)
        finally:
            sock.settimeout(prev_timeout)

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None
