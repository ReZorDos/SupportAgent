import argparse
import re
import shutil
import time
from pathlib import Path

from support_agent.settings import ROOT, runtime_directory


def expired_sessions(runtime, now=None, days=7):
    runtime = Path(runtime)
    now = time.time() if now is None else now
    if not runtime.exists():
        return []
    selected = []
    for path in runtime.iterdir():
        if (path.is_dir() and not path.is_symlink()
                and re.fullmatch(r"[0-9a-f]{32}", path.name)
                and now - path.stat().st_mtime > days * 86400):
            selected.append(path)
    return sorted(selected)


def purge(runtime, apply=False, now=None, days=7):
    selected = expired_sessions(runtime, now=now, days=days)
    if apply:
        for path in selected:
            shutil.rmtree(path)
    return selected


def expired_monitor_logs(runtime, now=None, days=7):
    directory = Path(runtime) / "monitor"
    if not directory.is_dir() or directory.is_symlink():
        return []
    now = time.time() if now is None else now
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and not path.is_symlink()
                  and re.fullmatch(r"(?:metrics|alerts)-\d{4}-\d{2}-\d{2}\.jsonl", path.name)
                  and now - path.stat().st_mtime > days * 86400)


def main():
    parser = argparse.ArgumentParser(description="Очистка старых журналов диалогов")
    parser.add_argument("--apply", action="store_true", help="Удалить выбранные папки; без флага только показать")
    args = parser.parse_args()
    runtime = runtime_directory(ROOT)
    selected = purge(runtime, apply=args.apply)
    monitor_files = expired_monitor_logs(runtime)
    if args.apply:
        for path in monitor_files:
            path.unlink()
    print("Удалено:" if args.apply else "Подлежат удалению после проверки:", len(selected))
    for path in selected:
        print(path)
    print("Файлы монитора:", len(monitor_files))
    for path in monitor_files:
        print(path)
    print("База памяти и папки eval_ не затрагиваются.")


if __name__ == "__main__":
    main()
