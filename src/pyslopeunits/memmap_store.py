from __future__ import annotations

from pathlib import Path
import gc
import json
import os
import time
from typing import Any

import numpy as np

try:  # Optional: use when available, but do not make it a hard dependency.
    import psutil  # type: ignore
except Exception:  # pragma: no cover - exercised only on systems without psutil
    psutil = None


class _RAMArray(np.ndarray):
    """ndarray with a memmap-compatible ``flush`` no-op.

    Numba accepts ndarray subclasses, so local temporary kernels can use this
    array exactly like a memmap while avoiding disk I/O when RAM is available.
    Persistent/shared arrays are deliberately *not* allocated this way because
    spawned workers must be able to reopen them by filename.
    """

    def __new__(cls, shape, dtype, *, name: str | None = None):
        obj = np.empty(shape, dtype=dtype).view(cls)
        obj._pyslope_name = name
        obj._pyslope_backend = "ram"
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._pyslope_name = getattr(obj, "_pyslope_name", None)
        self._pyslope_backend = getattr(obj, "_pyslope_backend", "ram")

    def flush(self) -> None:
        return None


def _physical_memory() -> tuple[int, int]:
    """Return ``(total, available)`` physical RAM in bytes.

    ``psutil`` is preferred when installed.  Native fallbacks keep the package
    usable on systems where psutil is absent.  If memory cannot be determined,
    ``(0, 0)`` is returned and temporary allocation safely falls back to disk.
    """

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:
            pass

    if os.name == "nt":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            state = MEMORYSTATUSEX()
            state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return int(state.ullTotalPhys), int(state.ullAvailPhys)
        except Exception:
            pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and total_pages > 0 and available_pages >= 0:
            return total_pages * page_size, available_pages * page_size
    except Exception:
        pass

    return 0, 0


class MemmapStore:
    """Hybrid array store used by PySlopeUnits.

    Storage policy
    --------------
    Persistent/shared arrays
        Always file-backed ``.npy`` memmaps.  This is required for
        multiprocessing ``spawn``, reusable caches and restart/checkpoint
        semantics.  The operating system may still keep their hot pages in RAM.

    Local expendable scratch
        RAM first when both the configured scratch budget and current free RAM
        permit it; otherwise transparently spill to a file-backed memmap.

    The numerical algorithm is independent of the selected storage backend.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        ram_budget_bytes: int | None = None,
        ram_fraction: float = 0.50,
        min_free_ram_bytes: int | None = None,
        verbose: bool = False,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.verbose = bool(verbose)

        if not (0.05 <= float(ram_fraction) <= 0.90):
            raise ValueError("ram_fraction must be between 0.05 and 0.90")

        total, available = _physical_memory()

        # If the caller supplies an execution budget (e.g. --memory-limit-gb),
        # scratch RAM is a fraction of that budget. Otherwise use installed RAM.
        # If RAM size cannot be detected, prefer the safe all-disk fallback.
        if ram_budget_bytes is None:
            capacity = total
        else:
            capacity = max(0, int(ram_budget_bytes))

        self.ram_budget_bytes = max(0, int(capacity * float(ram_fraction)))

        if min_free_ram_bytes is not None:
            self.min_free_ram_bytes = max(0, int(min_free_ram_bytes))
        elif total > 0:
            self.min_free_ram_bytes = int(
                max(384 * 1024**2, min(total // 10, 2 * 1024**3))
            )
        else:
            self.min_free_ram_bytes = 384 * 1024**2

        self._ram_arrays: dict[str, np.ndarray] = {}
        self._ram_bytes = 0
        self._disk_temp_bytes = 0
        self._allocation_log: list[dict[str, Any]] = []

        # Do not reserve a theoretical scratch budget that is already
        # unavailable at store creation. Unknown availability -> all disk.
        if available > 0:
            available_for_scratch = max(0, available - self.min_free_ram_bytes)
            self.ram_budget_bytes = min(self.ram_budget_bytes, available_for_scratch)
        else:
            self.ram_budget_bytes = 0

    def path(self, name: str) -> Path:
        return self.root / f"{name}.npy"

    @staticmethod
    def _nbytes(shape, dtype) -> int:
        return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)

    def _can_allocate_ram(self, nbytes: int) -> bool:
        if nbytes <= 0:
            return True
        if self._ram_bytes + nbytes > self.ram_budget_bytes:
            return False

        _, available = _physical_memory()
        if available <= 0:
            return False
        return available - nbytes >= self.min_free_ram_bytes

    def _log_alloc(self, name: str, backend: str, nbytes: int) -> None:
        self._allocation_log.append(
            {"name": name, "backend": backend, "bytes": int(nbytes)}
        )
        if self.verbose:
            print(
                f"[PySlopeUnits memory] {name}: {backend.upper()} | "
                f"{nbytes / 1024**2:.1f} MB | "
                f"scratch-RAM={self._ram_bytes / 1024**2:.1f}/"
                f"{self.ram_budget_bytes / 1024**2:.1f} MB"
            )

    def create(self, name: str, shape, dtype, *, fill=None, mode="w+"):
        """Create a persistent/shared disk-backed ``.npy`` memmap."""
        self._drop_ram(name)
        p = self.path(name)
        arr = np.lib.format.open_memmap(p, mode=mode, dtype=dtype, shape=shape)
        if fill is not None:
            arr[...] = fill
            arr.flush()
        return arr

    def create_shared(self, name: str, shape, dtype, *, fill=None, mode="w+"):
        """Explicit alias for arrays that spawned workers must reopen."""
        return self.create(name, shape, dtype, fill=fill, mode=mode)

    def create_temp(
        self,
        name: str,
        shape,
        dtype,
        *,
        fill=None,
        prefer_ram: bool = True,
    ):
        """Create local expendable scratch in RAM, spilling to disk if needed."""
        self.remove(name, best_effort=True)
        nbytes = self._nbytes(shape, dtype)

        if prefer_ram and self._can_allocate_ram(nbytes):
            try:
                arr = _RAMArray(shape, dtype, name=name)
                if fill is not None:
                    arr[...] = fill
                self._ram_arrays[name] = arr
                self._ram_bytes += nbytes
                self._log_alloc(name, "ram", nbytes)
                return arr
            except (MemoryError, OSError):
                gc.collect()
                # Free RAM can change between the check and allocation.

        p = self.path(name)
        arr = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape)
        if fill is not None:
            arr[...] = fill
            arr.flush()
        self._disk_temp_bytes += nbytes
        self._log_alloc(name, "disk", nbytes)
        return arr

    # Compatibility alias for code that names scratch explicitly.
    create_scratch = create_temp

    def open(self, name: str, mode="r"):
        arr = self._ram_arrays.get(name)
        if arr is not None:
            return arr
        return np.load(self.path(name), mmap_mode=mode, allow_pickle=False)

    def exists(self, name: str) -> bool:
        return name in self._ram_arrays or self.path(name).exists()

    def backend(self, name_or_array) -> str:
        if isinstance(name_or_array, str):
            if name_or_array in self._ram_arrays:
                return "ram"
            return "disk" if self.path(name_or_array).exists() else "missing"
        if isinstance(name_or_array, _RAMArray):
            return "ram"
        return "disk" if isinstance(name_or_array, np.memmap) else "array"

    @staticmethod
    def close_array(arr) -> None:
        """Flush/close memmaps; RAM arrays require no explicit close."""
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
        for arr in arrays:
            cls.close_array(arr)

    def _drop_ram(self, name: str) -> bool:
        arr = self._ram_arrays.pop(name, None)
        if arr is None:
            return False
        self._ram_bytes = max(0, self._ram_bytes - int(arr.nbytes))
        return True

    def release_temp(self, name_or_array, *, remove_disk: bool = True) -> bool:
        """Release one local temporary allocation and its scratch budget."""
        if isinstance(name_or_array, str):
            name = name_or_array
        else:
            name = getattr(name_or_array, "_pyslope_name", None)
            if name is None:
                self.close_array(name_or_array)
                return False

        if name in self._ram_arrays:
            return self._drop_ram(name)

        p = self.path(name)
        if p.exists() and remove_disk:
            return self.remove(name, best_effort=True)
        return p.exists()

    def cleanup(self, name: str) -> bool:
        return self.remove(name, best_effort=True)

    def remove(
        self,
        name: str,
        *,
        best_effort: bool = False,
        retries: int = 8,
        delay: float = 0.20,
    ) -> bool:
        """Remove a RAM temporary or disk store file."""
        removed_ram = self._drop_ram(name)
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
            return bool(removed_ram)
        p.unlink()
        return True

    def memory_report(self) -> dict[str, Any]:
        return {
            "scratch_ram_budget_bytes": int(self.ram_budget_bytes),
            "scratch_ram_bytes_current": int(self._ram_bytes),
            "disk_temp_bytes_allocated": int(self._disk_temp_bytes),
            "allocations": list(self._allocation_log),
        }

    def write_memory_report(self, name: str = "memory_allocation_report.json") -> Path:
        return self.write_json(name, self.memory_report())

    def write_json(self, name: str, obj) -> Path:
        p = self.root / name
        p.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        return p
