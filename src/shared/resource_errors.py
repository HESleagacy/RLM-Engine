"""Stable resource errors without guessing why the operating system killed a worker."""

from __future__ import annotations

import errno
import sqlite3
from functools import wraps


class ResourceExhaustedError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class WorkerTerminatedError(RuntimeError):
    code = "worker_terminated"


DISK_FULL = "Disk space or storage quota exhausted. Free space or choose another storage directory."
OUT_OF_MEMORY = "Available memory exhausted. Close other workloads or use a smaller input/read size."


def resource_error(exc: BaseException) -> ResourceExhaustedError | None:
    """Recognize wrapped allocation/storage failures, not ordinary I/O errors."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ResourceExhaustedError):
            return exc
        if isinstance(exc, MemoryError):
            return ResourceExhaustedError("out_of_memory", OUT_OF_MEMORY)
        if isinstance(exc, sqlite3.Error):
            code = getattr(exc, "sqlite_errorcode", None)
            # Python 3.10 does not expose sqlite_errorcode.
            message = str(exc).lower()
            if (code is not None and code & 255 == sqlite3.SQLITE_FULL) or message == "database or disk is full":
                return ResourceExhaustedError("disk_full", DISK_FULL)
            if (code is not None and code & 255 == sqlite3.SQLITE_NOMEM) or message == "out of memory":
                return ResourceExhaustedError("out_of_memory", OUT_OF_MEMORY)
        if isinstance(exc, OSError):
            if exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)):
                return ResourceExhaustedError("disk_full", DISK_FULL)
            if exc.errno == errno.ENOMEM:
                return ResourceExhaustedError("out_of_memory", OUT_OF_MEMORY)
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return None


def report_resource_errors(function):
    """Translate errors at public boundaries while preserving unrelated failures."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            failure = resource_error(exc)
            if failure is None or failure is exc:
                raise
            raise failure from exc
    return wrapped
