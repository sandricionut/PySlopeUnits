from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import tempfile
from typing import Any


CHECKPOINT_SCHEMA_VERSION = 1


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON checkpoint with an atomic rename.

    The file is written in the destination directory, flushed and fsynced, then
    replaced atomically.  A crash can therefore leave either the previous valid
    checkpoint or the new valid checkpoint, never a half-written JSON document.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
        # Best-effort fsync of the directory entry on POSIX.
        if os.name != "nt":
            try:
                dfd = os.open(path.parent, os.O_DIRECTORY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except Exception:
                pass
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def read_json(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def path_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    info: dict[str, Any] = {"path": str(p.resolve()) if p.exists() else str(p)}
    try:
        st = p.stat()
        info.update({"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    except Exception:
        pass
    return info


@dataclass(frozen=True)
class CheckpointPolicy:
    enabled: bool = True
    interval_minutes: float = 15.0
    restart: bool = False

    def __post_init__(self):
        if self.interval_minutes <= 0:
            raise ValueError("checkpoint interval must be > 0 minutes")
