import json
import math
import uuid
from .settings import ROOT, runtime_directory
from .telemetry import Journal
from .llm import Ollama
from .workflow import build_graph, run_turn
from .memory import MemoryStore
from .single_process import single_process_lock


def read_metadata(filename="", size=None):
    old_name = f" (Enter — оставить {filename})" if filename else " (Enter — неизвестно)"
    entered_name = input("Имя файла, если известно" + old_name + ": ").strip()
    if entered_name:
        filename = entered_name
    old_size = f" (Enter — оставить {size} МБ)" if size is not None else " (Enter — неизвестно)"
    entered_size = input("Размер в МБ, если известен" + old_size + ": ").strip()
    if entered_size:
        size = float(entered_size.replace(",", "."))
    if size is not None and (not math.isfinite(size) or size < 0):
        raise ValueError("Размер должен быть конечным неотрицательным числом")
    return filename, size

def run_main():
    print("Учебный симулятор поддержки")
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    journal = Journal(runtime_directory(ROOT) / uuid.uuid4().hex)
    client = Ollama(config, journal)
    memory_store = None
    try:
        client.prepare()
        memory_store = MemoryStore(runtime_directory(ROOT) / "memory.sqlite3")
        for category in ("upload", "service", "access", "login", "materials", "grades"):
            memory_store.sync_local_procedure(category, ROOT / "skills" / f"{category}.md")
        graph = build_graph(client, journal, config, memory_store)
        history, filename, size = [], "", None
        course_code = demo_account_code = material_code = submission_code = ""
        last_result = None
        print("Для завершения введите /exit. Если рекомендация помогла, введите /resolved.")
        while True:
            message = input("Обращение: " if not history else "Следующее сообщение: ").strip()
            if message.lower() in {"/exit", "выход"}:
                break
            if message.lower() == "/resolved":
                if not last_result or last_result.get("status") != "solution":
                    print("Сначала нужно получить решение в текущем диалоге.")
                    continue
                episode = memory_store.confirm(last_result)
                journal.turn_id = last_result.get("trace_turn_id")
                try:
                    journal.emit("memory_confirmed", episode=episode)
                finally:
                    journal.turn_id = None
                last_result = None
                print("Спасибо, успешный исход сохранён в локальной памяти симулятора.")
                continue
            if not message or len(message) > 1500:
                print("Введите сообщение от 1 до 1500 символов")
                continue
            filename, size = read_metadata(filename, size)
            entered_course = input("Код курса, если известен (Enter — оставить прежний или неизвестен): ").strip()
            if entered_course:
                course_code = entered_course[:30].upper()
            entered_account = input("Код тестового профиля, если известен (Enter — оставить прежний или неизвестен): ").strip()
            if entered_account:
                demo_account_code = entered_account[:30].upper()
            entered_material = input("Код тестового материала, если известен (Enter — оставить прежний или неизвестен): ").strip()
            if entered_material:
                material_code = entered_material[:30].upper()
            entered_submission = input("Код тестовой сдачи, если известен (Enter — оставить прежний или неизвестен): ").strip()
            if entered_submission:
                submission_code = entered_submission[:30].upper()
            result = run_turn(graph, journal, message, filename, size, history, course_code,
                              demo_account_code, material_code, submission_code)
            last_result = result
            turn = len(history) // 2 + 1
            (journal.directory / f"turn-{turn:02d}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            history.extend([{"role": "user", "content": message, "filename": filename, "size_mb": size,
                             "course_code": course_code, "demo_account_code": demo_account_code,
                             "material_code": material_code, "submission_code": submission_code},
                            {"role": "assistant", "content": result["final"], "status": result["status"]}])
            (journal.directory / "conversation.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
            print("\nОтвет:", result["final"])
            print("Статус:", result["status"])
            if result["status"] == "solution":
                print("Если рекомендация помогла, введите /resolved вместо следующего сообщения.")
    except Exception as exc:
        journal.emit("error", message=str(exc))
        print("Процесс остановлен без выдачи непроверенного ответа:", exc)
        raise SystemExit(1)
    finally:
        if memory_store:
            memory_store.close()
        try:
            client.close()
        except Exception as exc:
            journal.emit("model_unload_error", role="shutdown", message=str(exc))
            print("Выгрузка моделей не подтверждена:", exc)
        print("Файлы обращения:", journal.directory)

def main():
    with single_process_lock(runtime_directory(ROOT) / "process.lock"):
        run_main()
