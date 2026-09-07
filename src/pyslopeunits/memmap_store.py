from __future__ import annotations

from pathlib import Path
import json
import gc
import time
import numpy as np


class MemmapStore:
    """Small file-system store for .npy memory-mapped arrays."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.npy"

    def create(self, name: str, shape, dtype, *, fill=None, mode="w+"):
        p = self.path(name)
        arr = np.lib.format.open_memmap(p, mode=mode, dtype=dtype, shape=shape)
        if fill is not None:
            arr[...] = fill
            arr.flush()
        return arr

    def open(self, name: str, mode="r"):
        return np.load(self.path(name), mmap_mode=mode, allow_pickle=False)

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    @staticmethod
    def close_array(arr) -> None:
        """Flush and explicitly close a NumPy memmap, especially on Windows."""
        if arr is None:
            return
        try:
            arr.flush()
        except Exception:
            pass
        mm = getattr(arr, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass

    @classmethod
    def close_many(cls, *arrays) -> None:
        """Explicitly close several NumPy memmaps."""
        for arr in arrays:
            cls.close_array(arr)

    def cleanup(self, name: str) -> bool:
        """Best-effort cleanup for an expendable temporary memmap.

        Cleanup must never invalidate a completed computation on Windows.
        """
        return self.remove(name, best_effort=True)

    def remove(
        self,
        name: str,
        *,
        best_effort: bool = False,
        retries: int = 8,
        delay: float = 0.20,
    ) -> bool:
        """Remove a store file.

        With ``best_effort=True`` a Windows file-handle cleanup problem cannot
        invalidate an otherwise completed computation. The stale temporary file
        can be removed on the next process start.
        """
        p = self.path(name)
        if not p.exists():
            return True

        for _ in range(max(1, int(retries))):
            try:
                p.unlink()
                return True
            except FileNotFoundError:
                return True
            except PermissionError:
                gc.collect()
                time.sleep(float(delay))

        if best_effort:
            return False
        p.unlink()
        return True

    def write_json(self, name: str, obj) -> Path:
        p = self.root / name
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return p
