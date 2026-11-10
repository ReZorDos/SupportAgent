import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_trace(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    turns = defaultdict(list)
    for row in rows:
        turns[row.get("turn_id") or "без_хода"].append(row)
    return rows[0].get("trace_id", "старый_журнал") if rows else "пустой_журнал", turns


def summarize_turn(turn_id, rows):
    starts = {row["span_id"]: row for row in rows if row["event"] == "span_start"}
    spans = []
    for row in rows:
        if row["event"] == "span_end":
            start = starts.get(row["span_id"], {})
            spans.append({"kind": row["kind"], "name": row["name"],
                          "parent": start.get("parent_span_id"), "status": row["status"],
                          "seconds": row["duration_seconds"],
                          "model": start.get("model"), "prompt_version": start.get("prompt_version")})
    trajectory = []
    for row in rows:
        event = row["event"]
        if event == "route":
            trajectory.append("маршрут: " + row["category"])
        elif event == "memory_lookup":
            trajectory.append("память: " + str(row["confirmed_count"]) + " подтверждённых случаев")
        elif event == "diagnostic_decision":
            trajectory.append("диагност: " + row["action"])
        elif event == "procedure_loaded":
            trajectory.append("процедура: " + row["category"])
        elif event == "tool_result":
            trajectory.append("инструмент: " + row["observation"]["id"])
        elif event == "clarification_fallback":
            trajectory.append("уточнение: резервный вопрос по результату инструмента")
        elif event == "solution_fallback":
            trajectory.append("решение: резервный текст по результату инструмента")
        elif event == "validation":
            trajectory.append("валидация: " + ("принято" if row["approved"] else "отклонено"))
        elif event == "judge_disagreement":
            trajectory.append("расхождение валидатора: принят проверенный шаблон")
        elif event == "model_residency":
            trajectory.append("модели в памяти: " + ", ".join(row["models"]))
        elif event == "model_switch":
            trajectory.append("выгружена модель: " + row["unloaded"])
        elif event == "turn_end":
            trajectory.append("исход: " + row["outcome"])
        elif event == "memory_confirmed":
            trajectory.append("пользователь подтвердил результат")
        elif event in {"turn_error", "tool_error", "llm_error"}:
            trajectory.append("ошибка: " + event)
    return {"turn_id": turn_id, "trajectory": trajectory, "spans": spans}


def main():
    parser = argparse.ArgumentParser(description="Просмотр одного локального трейса")
    parser.add_argument("events", type=Path)
    args = parser.parse_args()
    trace_id, turns = load_trace(args.events)
    print("Trace ID:", trace_id)
    for turn_id, rows in turns.items():
        if turn_id == "без_хода" and not any(row["event"] == "route" for row in rows):
            continue
        view = summarize_turn(turn_id, rows)
        print("\nХод:", turn_id)
        print("Траектория:", " → ".join(view["trajectory"]))
        for span in view["spans"]:
            version = f", {span['model']}, промпт {span['prompt_version']}" if span["model"] else ""
            print(f"  {span['kind']}: {span['name']} — {span['seconds']} с, {span['status']}{version}")


if __name__ == "__main__":
    main()
