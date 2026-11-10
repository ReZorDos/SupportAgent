import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from support_agent.settings import ROOT, runtime_directory
from support_agent.system_metrics import snapshot


LOW_AVAILABLE_RAM_GB = 0.75
LOW_DISK_GB = 2.0
MIN_QUALITY_CASES = 5
MIN_CORRECT_MODE_RATE = 0.8


def ollama_healthy():
    client = build_opener(ProxyHandler({}))
    try:
        with client.open("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            return isinstance(json.load(response).get("models"), list)
    except (OSError, ValueError, KeyError):
        return False


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def daily_file(directory, category, timestamp):
    date = datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%d")
    return directory / f"{category}-{date}.jsonl"


def quality_below_target(runtime):
    paths = sorted(Path(runtime).glob("eval_*/eval_results.json"),
                   key=lambda path: path.stat().st_mtime, reverse=True)
    cases = []
    for path in paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if report.get("selected_case") != "oversize_pdf":
            continue
        cases.extend(row for row in report.get("cases", []) if row.get("status") and "error" not in row)
        if len(cases) >= 10:
            break
    cases = cases[:10]
    if len(cases) < MIN_QUALITY_CASES:
        return None
    correct = sum(row.get("checks", {}).get("mode_matches") is True for row in cases)
    return correct / len(cases) < MIN_CORRECT_MODE_RATE


def latest_eval_alerts(runtime, last_seen):
    candidates = sorted(runtime.glob("eval_*/eval_summary.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates or str(candidates[0]) == last_seen:
        return [], last_seen
    path = candidates[0]
    raw = path.with_name("alerts.jsonl")
    alerts = []
    if raw.exists():
        try:
            alerts = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, ValueError):
            return [], last_seen
    return [{"kind": "eval_" + item["kind"], "eval_dir": path.parent.name,
             "case": item.get("case")} for item in alerts], str(path)


def sample_once(runtime, previous_cpu=None, active=None, last_eval=None,
                sampler=snapshot, health=ollama_healthy):
    runtime = Path(runtime)
    output = runtime / "monitor"
    output.mkdir(parents=True, exist_ok=True)
    metrics, current_cpu = sampler(runtime, previous_cpu)
    metrics["ollama_healthy"] = bool(health())
    append_jsonl(daily_file(output, "metrics", metrics["timestamp"]), metrics)
    now = set()
    if not metrics["ollama_healthy"]:
        now.add("ollama_unavailable")
    if metrics["available_ram_gb"] < LOW_AVAILABLE_RAM_GB:
        now.add("low_available_ram")
    if metrics["disk_free_gb"] < LOW_DISK_GB:
        now.add("low_disk")
    if quality_below_target(runtime) is True:
        now.add("quality_below_target")
    active = set() if active is None else active
    alerts = [{"timestamp": metrics["timestamp"], "kind": kind, "state": "firing"}
              for kind in sorted(now - active)]
    alerts += [{"timestamp": metrics["timestamp"], "kind": kind, "state": "resolved"}
               for kind in sorted(active - now)]
    eval_alerts, last_eval = latest_eval_alerts(runtime, last_eval)
    alerts += [{"timestamp": metrics["timestamp"], "state": "firing", **item}
               for item in eval_alerts]
    for alert in alerts:
        append_jsonl(daily_file(output, "alerts", metrics["timestamp"]), alert)
        print("[ALERT]", json.dumps(alert, ensure_ascii=False), flush=True)
    return current_cpu, now, last_eval


def main():
    parser = argparse.ArgumentParser(description="Локальный монитор здоровья VM")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    args = parser.parse_args()
    if not 5 <= args.interval <= 3600:
        parser.error("Интервал должен быть от 5 до 3600 секунд")
    runtime = runtime_directory(ROOT)
    previous_cpu = None
    active = set()
    last_eval = None
    while True:
        try:
            previous_cpu, active, last_eval = sample_once(runtime, previous_cpu, active, last_eval)
        except Exception as exc:
            output = runtime / "monitor"
            output.mkdir(parents=True, exist_ok=True)
            now = time.time()
            append_jsonl(daily_file(output, "alerts", now), {"timestamp": now,
                                                               "kind": "monitor_error", "error": type(exc).__name__})
            print("[ALERT] monitor_error", type(exc).__name__, flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
