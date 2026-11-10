import ctypes
import os
import shutil
import time
from pathlib import Path


GIB = 1024 ** 3


def memory_snapshot(meminfo_path=Path("/proc/meminfo")):
    if os.name == "nt":
        class Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total", ctypes.c_ulonglong), ("available", ctypes.c_ulonglong),
                        ("swap_total", ctypes.c_ulonglong), ("swap_available", ctypes.c_ulonglong),
                        ("virtual_total", ctypes.c_ulonglong), ("virtual_available", ctypes.c_ulonglong),
                        ("extended", ctypes.c_ulonglong)]

        status = Status()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Не удалось получить данные об ОЗУ")
        total = status.total / GIB
        available = status.available / GIB
        return {"total_ram_gb": total, "used_ram_gb": total - available,
                "available_ram_gb": available, "swap_used_gb": None}

    entries = {}
    for line in meminfo_path.read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        entries[key] = int(value.strip().split()[0])
    total = entries["MemTotal"] * 1024 / GIB
    available = entries["MemAvailable"] * 1024 / GIB
    return {"total_ram_gb": total, "used_ram_gb": total - available,
            "available_ram_gb": available,
            "swap_used_gb": (entries["SwapTotal"] - entries["SwapFree"]) * 1024 / GIB}


def cpu_sample(stat_path=Path("/proc/stat")):
    fields = [int(part) for part in stat_path.read_text(encoding="ascii").splitlines()[0].split()[1:]]
    return sum(fields), fields[3] + fields[4]


def cpu_percent(first, second):
    total = second[0] - first[0]
    return round(100 * (1 - (second[1] - first[1]) / total), 2) if total > 0 else None


def snapshot(path, previous_cpu=None):
    memory = memory_snapshot()
    disk = shutil.disk_usage(path)
    current_cpu = cpu_sample() if Path("/proc/stat").exists() else None
    sample = {"timestamp": time.time(), **memory,
              "ram_percent": round(100 * memory["used_ram_gb"] / memory["total_ram_gb"], 2),
              "disk_free_gb": round(disk.free / GIB, 2),
              "cpu_percent": cpu_percent(previous_cpu, current_cpu)
              if previous_cpu and current_cpu else None}
    return sample, current_cpu
