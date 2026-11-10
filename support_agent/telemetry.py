import json
import time
import uuid
from pathlib import Path


class Journal:
    def __init__(self, directory, record_payloads=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.trace_id = uuid.uuid4().hex
        self.turn_id = None
        self.record_payloads = record_payloads

    def emit(self, event, **data):
        if event == "llm_result" and not self.record_payloads:
            response = data.pop("response", {})
            request = data.pop("request", {})
            data["response"] = {key: response.get(key) for key in (
                "done", "done_reason", "load_duration", "prompt_eval_count",
                "prompt_eval_duration", "eval_count", "eval_duration")}
            data["schema"] = request.get("format", {}).get("title")
        row = {"timestamp": time.time(), "trace_id": self.trace_id,
               "turn_id": self.turn_id, "event": event, **data}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if event in {"error", "llm_error", "model_unload_error", "tool_error", "turn_error",
                     "knowledge_expired", "evidence_conflict", "validator_retry_exhausted"}:
            alert = {"timestamp": row["timestamp"], "trace_id": self.trace_id,
                     "turn_id": self.turn_id, "kind": event,
                     "role": data.get("role"), "action": data.get("action")}
            with (self.directory / "alerts.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(alert, ensure_ascii=False) + "\n")
        if event in {"span_start", "span_end", "turn_end"}:
            return
        detail = ""
        if event == "agent_start":
            detail = f" {data.get('role')} / {data.get('model')}"
        elif event == "validation":
            detail = f" approved={data.get('approved')}; " + "; ".join(data.get("issues", []))
        elif event == "diagnostic_decision":
            detail = " " + data.get("action", "")
        elif event == "evidence_binding":
            detail = " доказательства инструментов прикреплены к ответу"
        print(f"[{event}]{detail}", flush=True)
