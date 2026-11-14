"""Private, content-addressed conversion artifacts with atomic publication."""

from __future__ import annotations

import hashlib
import errno
import json
from pathlib import Path

from document_ingestion.text_extractor import PDFIngestionError


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65_536), b""):
            digest.update(block)
    return digest.hexdigest()


def source_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    @staticmethod
    def key(settings: dict) -> str:
        return hashlib.sha256(
            json.dumps(settings, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()

    def completed(self, key: str) -> dict | None:
        directory = self.root / key
        if not directory.exists():
            return None
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if (
                manifest["artifact_key"] != key
                or not isinstance(manifest["page_count"], int)
                or manifest["page_count"] <= 0
                or manifest["text_sha256"] != file_digest(directory / "content.txt")
            ):
                raise ValueError("artifact integrity check failed")
            return manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PDFIngestionError(
                f"Invalid cached artifact at {directory}; remove that artifact and retry"
            ) from exc

    def publish(self, temporary: Path, key: str) -> dict:
        try:
            temporary.rename(self.root / key)
        except OSError as exc:
            # A concurrent conversion may have published the same complete artifact.
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            if self.completed(key) is None:
                raise
        manifest = self.completed(key)
        if manifest is None:
            raise PDFIngestionError("Artifact publication failed")
        return manifest
