import argparse
import json
import math
from pathlib import Path


SLOW_CASE_SECONDS = 180
HIGH_SYSTEM_RAM_GB = 14.0
LOW_AVAILABLE_RAM_GB = 0.75
HIGH_RAM_PERCENT = 90.0


def percentile_nearest_rank(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * fraction) - 1)], 2)


def summarize(report):
    cases = report.get("cases", [])
    completed = [row for row in cases if row.get("status") is not None and "error" not in row]
    times = [row["wall_seconds"] for row in completed if isinstance(row.get("wall_seconds"), (int, float))]
    peaks = [row["peak_system_ram_gb"] for row in completed
             if isinstance(row.get("peak_system_ram_gb"), (int, float))]
    available = [row["min_available_ram_gb"] for row in completed
                 if isinstance(row.get("min_available_ram_gb"), (int, float))]
    swap = [row["peak_swap_used_gb"] for row in completed
            if isinstance(row.get("peak_swap_used_gb"), (int, float))]
    percentages = [100 * row["peak_system_ram_gb"] / row["total_system_ram_gb"]
                   for row in completed if isinstance(row.get("peak_system_ram_gb"), (int, float))
                   and isinstance(row.get("total_system_ram_gb"), (int, float))
                   and row["total_system_ram_gb"] > 0]
    marked = [row for row in completed if isinstance(row.get("manual_grade"), bool)]
    by_case = {}
    for row in completed:
        case_id = row.get("case", {}).get("id", "unknown")
        item = by_case.setdefault(case_id, {"runs": 0, "correct_mode": 0, "times": []})
        item["runs"] += 1
        item["correct_mode"] += row.get("status") == row.get("case", {}).get("expected")
        if isinstance(row.get("wall_seconds"), (int, float)):
            item["times"].append(row["wall_seconds"])
    repeat_spread = {case_id: {"runs": item["runs"], "correct_mode": item["correct_mode"],
                               "min_seconds": round(min(item["times"]), 2) if item["times"] else None,
                               "max_seconds": round(max(item["times"]), 2) if item["times"] else None}
                     for case_id, item in by_case.items()}
    def total_if_recorded(field):
        return sum(row[field] for row in completed) if completed and all(field in row for row in completed) else None

    summary = {
        "cases_planned": report.get("planned_count", len(cases)),
        "cases_completed": len(completed),
        "cases_failed_to_run": len([row for row in cases if "error" in row]),
        "mode_matches": sum(row.get("checks", {}).get("mode_matches") is True for row in completed),
        "all_automatic_checks_pass": sum(bool(row.get("checks")) and
                                         all(row["checks"].values()) for row in completed),
        "validator_approved": sum(row.get("checks", {}).get("validator_approved") is True
                                  for row in completed),
        "by_expected": {mode: {"completed": sum(row.get("case", {}).get("expected") == mode
                                                 for row in completed),
                                "correct_mode": sum(row.get("case", {}).get("expected") == mode
                                                    and row.get("status") == mode for row in completed)}
                        for mode in ("solution", "clarification", "escalation")},
        "repeat_spread": repeat_spread,
        "manually_reviewed": len(marked),
        "manually_accepted": sum(row["manual_grade"] is True for row in marked),
        "llm_calls": sum(row.get("llm_calls", 0) for row in completed),
        "prompt_tokens": total_if_recorded("prompt_tokens"),
        "generated_tokens": total_if_recorded("generated_tokens"),
        "model_load_seconds": round(total_if_recorded("model_load_seconds"), 2)
        if total_if_recorded("model_load_seconds") is not None else None,
        "validation_retries": sum(max(0, row.get("validation_attempts", 0) - 1)
                                  for row in completed),
        "judge_disagreements": sum(row.get("judge_disagreements", 0) for row in completed),
        "latency_p50_seconds": percentile_nearest_rank(times, 0.50),
        "latency_p95_seconds": percentile_nearest_rank(times, 0.95),
        "peak_system_ram_gb": round(max(peaks), 2) if peaks else None,
        "peak_system_ram_percent": round(max(percentages), 2) if percentages else None,
        "min_available_ram_gb": round(min(available), 2) if available else None,
        "peak_swap_used_gb": round(max(swap), 2) if swap else None,
        "limits": {"slow_case_seconds": SLOW_CASE_SECONDS,
                   "high_system_ram_gb_legacy": HIGH_SYSTEM_RAM_GB,
                   "low_available_ram_gb": LOW_AVAILABLE_RAM_GB,
                   "high_ram_percent": HIGH_RAM_PERCENT},
        "limitations": "Один прогон не доказывает надёжность; автоматические флаги не оценивают смысл ответа. ОЗУ измерена внутри запускающей ОС; для VM память хоста измеряется отдельно.",
    }
    alerts = []
    for row in cases:
        name = row.get("case", {}).get("id", "unknown")
        if row.get("error"):
            alerts.append({"case": name, "kind": "execution_error"})
        if row.get("status") is None:
            continue
        if row.get("checks") and not all(row["checks"].values()):
            alerts.append({"case": name, "kind": "automatic_check_failed",
                           "checks": [key for key, passed in row["checks"].items() if not passed]})
        if row.get("wall_seconds", 0) > SLOW_CASE_SECONDS:
            alerts.append({"case": name, "kind": "slow_case", "seconds": row["wall_seconds"]})
        peak = row.get("peak_system_ram_gb")
        total = row.get("total_system_ram_gb")
        if peak is not None and ((total and peak / total * 100 > HIGH_RAM_PERCENT)
                                 or (not total and peak > HIGH_SYSTEM_RAM_GB)):
            alerts.append({"case": name, "kind": "high_system_ram", "gb": peak})
        if (row.get("min_available_ram_gb") is not None
                and row["min_available_ram_gb"] < LOW_AVAILABLE_RAM_GB):
            alerts.append({"case": name, "kind": "low_available_ram",
                           "gb": row["min_available_ram_gb"]})
        if row.get("validation_attempts", 0) > 2:
            alerts.append({"case": name, "kind": "validator_retry_budget"})
        if row.get("judge_disagreements", 0):
            alerts.append({"case": name, "kind": "judge_disagreement",
                           "count": row["judge_disagreements"]})
    return summary, alerts


def markdown_report(report, summary, alerts):
    def display(value, suffix=""):
        return "нет данных" if value is None else f"{value}{suffix}"

    lines = ["# Сводка контрольного прогона", "",
             f"Начало (UTC): {report.get('started_at', 'неизвестно')}", "",
             f"Набор: {report.get('suite', 'старый')}; повторов: {report.get('repeats', 1)}; "
             f"отпечаток входных файлов: {report.get('fingerprint', 'нет данных')}.", "",
             f"Завершено: {summary['cases_completed']}/{summary['cases_planned']}; "
             f"режим совпал: {summary['mode_matches']}/{summary['cases_completed']}; "
             f"все узкие автопроверки пройдены: {summary['all_automatic_checks_pass']}/{summary['cases_completed']}.",
             f"Ручная оценка: {summary['manually_reviewed']} проверено, "
             f"{summary['manually_accepted']} принято. Число вызовов моделей: {summary['llm_calls']}; "
             f"повторов валидации: {summary['validation_retries']}; "
             f"расхождений с локальной политикой: {summary['judge_disagreements']}.",
             f"Токены: входные {display(summary['prompt_tokens'])}, "
             f"сгенерированные {display(summary['generated_tokens'])}; "
             f"суммарная загрузка моделей {display(summary['model_load_seconds'], ' с')}.",
             f"Задержка p50/p95: {summary['latency_p50_seconds']}/{summary['latency_p95_seconds']} с "
             "(метод ближайшего ранга).",
             f"Пик ОЗУ запускающей ОС: {display(summary['peak_system_ram_gb'], ' ГиБ')}; "
             f"занято {display(summary['peak_system_ram_percent'], '%')}; "
             f"минимум доступно {display(summary['min_available_ram_gb'], ' ГиБ')}; "
             f"пик swap {display(summary['peak_swap_used_gb'], ' ГиБ')}.", "",
             "| Сценарий | Повтор | Ожидалось | Получено | Автопроверки | Время, с |",
             "| --- | ---: | --- | --- | --- | ---: |"]
    for row in report.get("cases", []):
        checks = row.get("checks")
        check_status = ("пройдены" if all(checks.values()) else "ошибки: " +
                        ", ".join(k for k, passed in checks.items() if not passed)) if checks else "нет результата"
        lines.append(f"| {row.get('case', {}).get('id', '?')} | {row.get('repeat', 1)} | "
                     f"{row.get('case', {}).get('expected', '?')} | {row.get('status', 'ошибка')} | "
                     f"{check_status} | {row.get('wall_seconds', '—')} |")
    lines += ["", "Срез по ожидаемым режимам (совпадение/завершено): " + ", ".join(
        f"{mode} {item['correct_mode']}/{item['completed']}" for mode, item in summary["by_expected"].items()),
              "", f"Локальные сигналы: {len(alerts)}.", "", summary["limitations"],
              "Ответы следует проверить вручную в eval_results.json; режимы и узкие проверки не заменяют экспертную оценку.", ""]
    return "\n".join(lines)


def save_summary(source):
    source = Path(source)
    report = json.loads(source.read_text(encoding="utf-8"))
    summary, alerts = summarize(report)
    parent = source.parent
    (parent / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (parent / "eval_summary.md").write_text(markdown_report(report, summary, alerts), encoding="utf-8")
    (parent / "alerts.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in alerts), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Сводка одного контрольного прогона")
    parser.add_argument("result_file", type=Path)
    args = parser.parse_args()
    summary = save_summary(args.result_file)
    print("Готово:", args.result_file.parent / "eval_summary.md")
    print("Совпадений режимов:", summary["mode_matches"], "/", summary["cases_completed"])


if __name__ == "__main__":
    main()
