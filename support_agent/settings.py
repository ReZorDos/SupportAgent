import ctypes
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def runtime_directory(root=ROOT):
    return Path(os.environ.get("SUPPORT_RUNTIME_DIR", str(root / "runtime"))).expanduser().resolve()

def available_system_ram_gb():
    if not hasattr(ctypes, "windll"):
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("Не удалось измерить доступную физическую память")
    return status.ullAvailPhys / (1024 ** 3)
