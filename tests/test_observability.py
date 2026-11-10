import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from support_agent.telemetry import Journal
from support_agent.workflow import build_graph, run_turn
from retention import purge
from trace_view import load_trace, summarize_turn


class ServiceClient:
    def ask(self, role, schema, data):
        if role == "dispatcher":
            return {"category": "service", "reason": "test"}
        if role == "diagnostician":
            return {"action": "finish" if data["evidence"] else "get_service_status", "reason": "test"}
        if role == "resolver":
            return {"mode": "clarification", "answer": "Какой текст ошибки?", "evidence_ids": ["get_service_status"]}
        return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}


class ObservabilityTests(unittest.TestCase):
    def test_routine_llm_event_omits_prompt_and_generated_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(tmp)
            journal.emit("llm_result", role="resolver", model="demo", wall_seconds=1.5,
                         request={"format": {"title": "Draft"}, "messages": [{"content": "secret input"}]},
                         response={"done": True, "done_reason": "stop", "prompt_eval_count": 12,
                                   "eval_count": 8, "message": {"content": "secret output"}})
            raw = (Path(tmp) / "events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("secret input", raw)
            self.assertNotIn("secret output", raw)
            self.assertIn('"prompt_eval_count": 12', raw)

    def test_turn_memory_and_tool_spans_have_trace_parent_and_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(tmp)
            graph = build_graph(ServiceClient(), journal, {"max_tool_calls": 3, "max_revisions": 2})
            result = run_turn(graph, journal, "Учебный сайт не открывается", "", None, [])
            self.assertEqual(result["status"], "clarification")
            trace_id, turns = load_trace(Path(tmp) / "events.jsonl")
            self.assertEqual(trace_id, journal.trace_id)
            self.assertEqual(len(turns), 1)
            turn_id, rows = next(iter(turns.items()))
            view = summarize_turn(turn_id, rows)
            self.assertIn("маршрут: service", view["trajectory"])
            self.assertIn("инструмент: get_service_status", view["trajectory"])
            self.assertIn("исход: clarification", view["trajectory"])
            journal.turn_id = turn_id
            journal.emit("memory_confirmed", episode={"issue_code": "test"})
            journal.turn_id = None
            _, updated = load_trace(Path(tmp) / "events.jsonl")
            self.assertIn("пользователь подтвердил результат",
                          summarize_turn(turn_id, updated[turn_id])["trajectory"])
            self.assertEqual({s["kind"] for s in view["spans"]}, {"turn", "memory", "tool"})
            for span in view["spans"]:
                self.assertGreaterEqual(span["seconds"], 0)
                if span["kind"] != "turn":
                    self.assertEqual(span["parent"], turn_id)

    def test_tool_error_creates_local_alert_and_error_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(tmp)
            graph = build_graph(ServiceClient(), journal, {"max_tool_calls": 3, "max_revisions": 2,
                                                          "simulator_service_status": "invalid"})
            result = run_turn(graph, journal, "Учебный сайт не открывается", "", None, [])
            self.assertEqual(result["status"], "escalation")
            self.assertTrue((Path(tmp) / "ticket.json").exists())
            alerts = [json.loads(row) for row in (Path(tmp) / "alerts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("tool_error", {row["kind"] for row in alerts})
            _, turns = load_trace(Path(tmp) / "events.jsonl")
            view = summarize_turn(*next(iter(turns.items())))
            self.assertTrue(any(s["kind"] == "tool" and s["status"] == "error" for s in view["spans"]))
            self.assertIn("исход: escalation", view["trajectory"])

    def test_retention_previews_and_only_removes_old_session_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Path(tmp)
            old = runtime / ("a" * 32)
            new = runtime / ("b" * 32)
            eval_dir = runtime / "eval_old"
            for path in (old, new, eval_dir):
                path.mkdir()
            database = runtime / "memory.sqlite3"
            database.write_text("safe", encoding="utf-8")
            now = time.time()
            os.utime(old, (now - 9 * 86400, now - 9 * 86400))
            os.utime(eval_dir, (now - 9 * 86400, now - 9 * 86400))
            self.assertEqual(purge(runtime, now=now), [old])
            self.assertTrue(old.exists())
            self.assertEqual(purge(runtime, apply=True, now=now), [old])
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertTrue(eval_dir.exists())
            self.assertTrue(database.exists())


if __name__ == "__main__":
    unittest.main()
