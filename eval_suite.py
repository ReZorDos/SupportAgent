import argparse
import hashlib
import json
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from support_agent.telemetry import Journal
from support_agent.llm import Ollama
from support_agent.settings import ROOT, runtime_directory
from support_agent.workflow import build_graph, run_turn
from eval_summary import save_summary
from support_agent.single_process import single_process_lock
from support_agent.system_metrics import memory_snapshot


CASES = [
    {"id": "oversize_pdf", "message": "Не могу загрузить работу. Ошибка «Превышен размер файла».",
     "filename": "lab.pdf", "size_mb": 28, "expected": "solution"},
    {"id": "missing_metadata", "message": "Не могу загрузить лабораторную работу.",
     "filename": "", "size_mb": None, "expected": "clarification"},
    {"id": "unknown_error", "message": "После выбора файла возникает неизвестная ошибка.",
     "filename": "lab.pdf", "size_mb": 10, "expected": "clarification"},
    {"id": "unsupported_format", "message": "Не удаётся прикрепить работу.",
     "filename": "lab.docx", "size_mb": 3, "expected": "solution"},
    {"id": "service_available", "message": "Учебный сайт не открывается.",
     "filename": "", "size_mb": None, "expected": "clarification", "simulator_service_status": "available"},
    {"id": "service_unavailable", "message": "Учебный сайт не открывается.",
     "filename": "", "size_mb": None, "expected": "solution", "simulator_service_status": "unavailable"},
    {"id": "course_missing", "message": "Не вижу курс и задание.",
     "filename": "", "size_mb": None, "course_code": "", "expected": "clarification"},
    {"id": "course_enrolled", "message": "Не вижу курс и задание.",
     "filename": "", "size_mb": None, "course_code": "CS101", "expected": "clarification"},
    {"id": "course_not_enrolled", "message": "Не вижу курс и задание.",
     "filename": "", "size_mb": None, "course_code": "CS202", "expected": "solution"},
    {"id": "login_missing", "message": "Не могу войти в учебный профиль.",
     "filename": "", "size_mb": None, "demo_account_code": "", "expected": "clarification"},
    {"id": "login_active", "message": "При попытке входа возникает ошибка.",
     "filename": "", "size_mb": None, "demo_account_code": "DEMO_OK", "expected": "clarification"},
    {"id": "login_locked", "message": "Не могу войти в учебный профиль.",
     "filename": "", "size_mb": None, "demo_account_code": "DEMO_LOCKED", "expected": "solution"},
    {"id": "material_missing", "message": "Не вижу материал лекции.",
     "filename": "", "size_mb": None, "material_code": "", "expected": "clarification"},
    {"id": "material_draft", "message": "Не могу открыть материал лекции.",
     "filename": "", "size_mb": None, "material_code": "MAT102", "expected": "solution"},
    {"id": "material_published", "message": "Не открывается учебный материал.",
     "filename": "", "size_mb": None, "material_code": "MAT101", "expected": "clarification"},
    {"id": "material_unknown", "message": "Не вижу материал лекции.",
     "filename": "", "size_mb": None, "material_code": "MAT999", "expected": "clarification"},
    {"id": "grade_missing", "message": "Не вижу оценку за работу.",
     "filename": "", "size_mb": None, "submission_code": "", "expected": "clarification"},
    {"id": "grade_pending", "message": "Почему нет оценки за сданную работу?",
     "filename": "", "size_mb": None, "submission_code": "SUB101", "expected": "solution"},
    {"id": "grade_unpublished", "message": "Оценка не отображается.",
     "filename": "", "size_mb": None, "submission_code": "SUB102", "expected": "solution"},
    {"id": "grade_published", "message": "Оценка не видна мне на странице.",
     "filename": "", "size_mb": None, "submission_code": "SUB103", "expected": "clarification"},
    {"id": "course_not_open", "message": "Не вижу материалы курса CS303.",
     "course_code": "CS303", "expected": "solution"},
    {"id": "login_service_down", "message": "Не получается войти в тестовый профиль.",
     "demo_account_code": "DEMO_OFFLINE", "expected": "solution"},
    {"id": "material_storage_down", "message": "Не могу открыть учебный материал.",
     "material_code": "MAT103", "expected": "solution"},
]


class MemorySampler:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.available_samples = []
        self.swap_samples = []
        self.total_ram_gb = None
        self.worker = None

    def __enter__(self):
        def sample():
            while not self.stop.is_set():
                try:
                    memory = memory_snapshot()
                    self.total_ram_gb = memory["total_ram_gb"]
                    self.samples.append(memory["used_ram_gb"])
                    self.available_samples.append(memory["available_ram_gb"])
                    if memory["swap_used_gb"] is not None:
                        self.swap_samples.append(memory["swap_used_gb"])
                except (OSError, KeyError, ValueError):
                    pass
                self.stop.wait(0.5)

        self.worker = threading.Thread(target=sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.worker:
            self.worker.join(timeout=2)


def automatic_checks(case, result):
    answer = result["final"]
    lowered = answer.lower()
    checks = {"mode_matches": result["status"] == case["expected"],
              "no_password_request": not bool(re.search(
                  r"\b(?:укажите|сообщите|пришлите|введите|отправьте|напишите)\b[^.!?]{0,70}"
                  r"\b(?:парол[ья]|код\s+подтверждения)\b", lowered))}
    if case["expected"] == "escalation":
        checks["local_escalation_only"] = result["status"] == "escalation" and "локальн" in lowered
    else:
        checks["validator_approved"] = result.get("verdict", {}).get("approved") is True
    if case["id"] == "oversize_pdf":
        checks.update({"mentions_limit_20": bool(re.search(r"20\s*мб", lowered)),
                       "no_other_format": not bool(re.search(r"друг(?:ом|ой|ие|их)\s+формат", lowered))})
    if case["id"] == "missing_metadata":
        checks["asks_for_information"] = bool(re.search(r"укаж|уточн|сообщ|како|сколько|размер|формат", lowered))
    if case["id"] == "unknown_error":
        checks["does_not_claim_cause"] = "причина ошибки" not in lowered
    if case["id"] == "service_available":
        checks["no_unconfirmed_outage"] = not bool(re.search(
            r"^(?:учебный\s+)?(?:сайт|сервис)\b[^.!?]{0,45}\b(?:недоступен|не\s+работает)\b",
            answer.strip(), re.IGNORECASE))
    if case["id"] in {"course_not_enrolled", "course_not_open"}:
        checks["mentions_simulator"] = bool(re.search(r"симулятор|тестов(?:ый|ого|ом)", lowered))
    if case["id"] == "course_enrolled":
        checks["does_not_reask_course"] = not bool(re.search(
            r"\b(?:уточните|укажите|сообщите|напишите)\b[^.!?]{0,65}"
            r"\b(?:какой\s+(?:именно\s+)?курс|код\s+курса|название\s+курса)\b",
            lowered))
    if case["id"] in {"login_locked", "login_service_down"}:
        checks["mentions_demo_scope"] = bool(re.search(r"симулятор|тестов", lowered))
    if case["id"] == "login_missing":
        checks["no_unverified_account_state"] = not bool(re.search(
            r"\b(?:заблокирован\w*|активен|активна|активный|активная|разблокирован\w*)\b", lowered))
    if case["id"].startswith("login_"):
        checks["no_unverified_course_claim"] = not bool(re.search(r"\b(?:зачислен\w*|курс\w*)\b", lowered))
    if case["id"] == "login_active":
        checks["does_not_reask_demo_code"] = not bool(re.search(
            r"\b(?:уточните|укажите|сообщите|напишите)\b[^.!?]{0,65}"
            r"\b(?:код\s+(?:тестового\s+)?(?:профиля|аккаунта)|"
            r"какой\s+(?:именно\s+)?(?:профиль|аккаунт)|"
            r"имя\s+(?:тестового\s+)?профиля)\b", lowered))
        checks["no_unverified_block"] = not bool(re.search(r"(?<!не )\bзаблокирован\w*\b", lowered))
    if case["id"].startswith(("material_", "grade_")):
        checks["no_real_platform_claim"] = not bool(re.search(r"\b(?:реальн\w*|настоящ\w*)\b[^.!?]{0,50}"
                                                      r"\b(?:проверил|проверена|опубликован\w*)\b", lowered))
        if case["expected"] == "solution":
            checks["mentions_demo_scope"] = bool(re.search(r"симулятор|тестов", lowered))

    if case.get("size_mb") is not None and case.get("filename", "").lower().endswith(".pdf") and case["size_mb"] > 20:
        checks["mentions_pdf_limit"] = bool(re.search(r"20\s*мб", lowered))
        checks["no_disallowed_format"] = not bool(re.search(r"друг(?:ом|ой|ие|их)\s+формат", lowered))
    if case.get("demo_account_code"):
        checks["no_course_claim_for_login"] = not bool(re.search(r"\b(?:зачислен\w*|курс\w*)\b", lowered))
        if case["expected"] == "solution":
            checks["login_demo_scope"] = bool(re.search(r"симулятор|тестов", lowered))
        if case["demo_account_code"] == "DEMO_OK":
            checks["no_false_login_block"] = not bool(re.search(r"(?<!не )\bзаблокирован\w*\b", lowered))
    if any(case.get(field) for field in ("course_code", "material_code", "submission_code")) and case["expected"] == "solution":
        checks["demo_scope"] = bool(re.search(r"симулятор|тестов", lowered))
    if case.get("material_code") == "MAT101" or case.get("submission_code") == "SUB103":
        checks["no_false_unpublished"] = not bool(re.search(r"\b(?:не\s+опубликован\w*|проверя\w*)\b", lowered))
    return checks


def summarize_events(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    calls = [row for row in rows if row["event"] == "llm_result"]
    return {"llm_calls": len(calls), "model_wall_seconds": round(sum(row["wall_seconds"] for row in calls), 2),
            "prompt_tokens": sum(row.get("response", {}).get("prompt_eval_count", 0) for row in calls),
            "generated_tokens": sum(row.get("response", {}).get("eval_count", 0) for row in calls),
            "model_load_seconds": round(sum(row.get("response", {}).get("load_duration", 0)
                                            for row in calls) / 1e9, 2),
            "by_role_seconds": {role: round(sum(row["wall_seconds"] for row in calls if row["role"] == role), 2)
                                for role in ("dispatcher", "diagnostician", "resolver", "validator")},
            "validation_attempts": sum(row["event"] == "validation" for row in rows),
            "judge_disagreements": sum(row["event"] == "judge_disagreement" for row in rows)}


def source_fingerprint():
    paths = [ROOT / name for name in ("app.py", "eval_suite.py", "eval_summary.py", "config.json",
                                      "PURPOSE.md", "BEHAVIOR.md", "MEMORY_POLICY.md", "TOOL_POLICY.md",
                                      "requirements.txt",
                                      "acceptance_cases.json")]
    paths += sorted((ROOT / "support_agent").glob("*.py"))
    paths += sorted((ROOT / "prompts").glob("*.md"))
    paths += sorted((ROOT / "skills").glob("*.md"))
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def load_cases(suite):
    cases = CASES if suite == "dev" else json.loads((ROOT / "acceptance_cases.json").read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Повторяются идентификаторы сценариев")
    if suite == "acceptance":
        modes = {mode: sum(case["expected"] == mode for case in cases)
                 for mode in ("solution", "clarification", "escalation")}
        if modes != {"solution": 20, "clarification": 5, "escalation": 5}:
            raise ValueError("Приёмочный набор должен содержать 20 решений, 5 уточнений и 5 эскалаций")
    return cases


def run_main():
    parser = argparse.ArgumentParser(description="Локальные контрольные сценарии поддержки")
    parser.add_argument("--suite", choices=("dev", "acceptance"), default="dev")
    parser.add_argument("--repeats", type=int, help="Число последовательных повторов (приёмка: 3 по умолчанию)")
    parser.add_argument("--resume", type=Path, help="Продолжить прерванный eval_results.json")
    parser.add_argument("--case", help="Запустить один сценарий вместо всего набора")
    parser.add_argument("--skip-if-busy", action="store_true", help="Пропустить фоновый прогон, если идёт диалог")
    args = parser.parse_args()
    all_cases = load_cases(args.suite)
    if args.case and args.case not in {item["id"] for item in all_cases}:
        parser.error("Неизвестный сценарий для выбранного набора: " + args.case)
    cases = [item for item in all_cases if not args.case or item["id"] == args.case]
    repeats = args.repeats if args.repeats is not None else (3 if args.suite == "acceptance" else 1)
    if not 1 <= repeats <= 3:
        parser.error("Число повторов должно быть от 1 до 3")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    fingerprint = source_fingerprint()
    if args.resume:
        output = args.resume.resolve()
        report = json.loads(output.read_text(encoding="utf-8"))
        if (report.get("suite") != args.suite or report.get("repeats") != repeats
                or report.get("selected_case") != args.case or report.get("fingerprint") != fingerprint
                or report.get("config") != config):
            parser.error("Нельзя продолжить: набор, параметры или код/промпты изменились")
        directory = output.parent
    else:
        directory = runtime_directory(ROOT) / ("eval_" + uuid.uuid4().hex)
        directory.mkdir(parents=True)
        output = directory / "eval_results.json"
        report = {"started_at": datetime.now(timezone.utc).isoformat(), "suite": args.suite,
                  "selected_case": args.case, "repeats": repeats, "fingerprint": fingerprint,
                  "planned_count": len(cases) * repeats,
                  "config": config,
                  "note": "Синтетические сценарии; автопроверки не заменяют ручную оценку ответа.", "cases": []}
    client = Ollama(config, Journal(directory))
    client.prepare()

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        save_summary(output)

    save()
    planned = [(case, repeat) for repeat in range(1, repeats + 1) for case in cases]
    completed = {(record["case"]["id"], record.get("repeat", 1)) for record in report["cases"]
                 if record.get("status") and not record.get("error")}
    try:
        for number, (case, repeat) in enumerate(planned, 1):
            if (case["id"], repeat) in completed:
                print(f"[{number}/{len(planned)}] {case['id']} / {repeat}: уже выполнен", flush=True)
                continue
            if source_fingerprint() != fingerprint:
                raise RuntimeError("Код или промпты изменились во время прогона; результаты не смешиваются")

            journal = Journal(directory / f"{case['id']}_r{repeat}_{uuid.uuid4().hex[:8]}", record_payloads=True)
            client.journal = journal
            record = {"case": case, "repeat": repeat, "manual_grade": None, "manual_notes": ""}
            report["cases"].append(record)
            print(f"[{number}/{len(planned)}] {case['id']} / {repeat}", flush=True)
            started = time.perf_counter()
            try:
                with MemorySampler() as memory:
                    scenario_config = {**config,
                                       "simulator_service_status": case.get("simulator_service_status", "available"),
                                       "simulator_tool_failures": case.get("simulator_tool_failures", [])}
                    result = run_turn(build_graph(client, journal, scenario_config), journal, case["message"],
                                      case.get("filename", ""), case.get("size_mb"), [], case.get("course_code", ""),
                                      case.get("demo_account_code", ""), case.get("material_code", ""),
                                      case.get("submission_code", ""))
                (journal.directory / "result.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                record.update(status=result["status"], answer=result["final"],
                              checks=automatic_checks(case, result),
                              peak_system_ram_gb=round(max(memory.samples), 2) if memory.samples else None,
                              baseline_system_ram_gb=round(memory.samples[0], 2) if memory.samples else None,
                              total_system_ram_gb=round(memory.total_ram_gb, 2) if memory.total_ram_gb else None,
                              min_available_ram_gb=round(min(memory.available_samples), 2)
                              if memory.available_samples else None,
                              peak_swap_used_gb=round(max(memory.swap_samples), 2)
                              if memory.swap_samples else None,
                              **summarize_events(journal.directory / "events.jsonl"))
            except Exception as exc:
                record["error"] = str(exc)
                journal.emit("eval_error", message=str(exc))
                raise
            finally:
                record["wall_seconds"] = round(time.perf_counter() - started, 2)
                save()
                print("  Результат:", record.get("status", "ошибка"),
                      "Время:", record["wall_seconds"], "с", flush=True)
    finally:
        client.close()
    print("Готово. Проверьте тексты ответов вручную:", output)


def main():
    if "--skip-if-busy" in sys.argv:
        try:
            with single_process_lock(runtime_directory(ROOT) / "process.lock"):
                run_main()
        except RuntimeError as exc:
            if "Уже запущен диалог или eval" not in str(exc):
                raise
            print("SKIP: диалог или eval уже выполняется")
        return
    with single_process_lock(runtime_directory(ROOT) / "process.lock"):
        run_main()


if __name__ == "__main__":
    main()
