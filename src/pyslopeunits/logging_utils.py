from __future__ import annotations

from datetime import datetime
import threading

_PRINT_LOCK = threading.Lock()

def log(*args, **kwargs):
    """Timestamped, flushed process log output in local time."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    kwargs.setdefault("flush", True)
    with _PRINT_LOCK:
        print(f"[{timestamp}]", *args, **kwargs)
