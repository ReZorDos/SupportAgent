import sqlite3
import tempfile
import unittest
from pathlib import Path

from support_agent.telemetry import Journal
from support_agent.workflow import build_graph, run_turn
from support_agent.memory import MemoryStore


class MemoryTests(unittest.TestCase):
    def test_confirmed_episode_survives_restart_without_raw_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            result = {"status": "solution", "category": "upload", "filename": "my_secret_name.pdf",
                      "message": "Пароль secret", "final": "Сожмите файл", "verdict": {"approved": True},
                      "evidence": [{"id": "check_file", "data": {"errors": ["file_too_large"]}}]}
            memory.confirm(result)
            memory.close()
            memory = MemoryStore(path)
            context = memory.lookup("upload", "another.pdf")
            self.assertEqual(context["confirmed_episodes"][0]["resolution_kind"], "compress_pdf")
            self.assertEqual(memory.lookup("upload", "another.docx")["confirmed_episodes"], [])
            raw = memory.connection.execute("SELECT * FROM episodes").fetchall()
            self.assertNotIn("my_secret_name", repr(raw))
            self.assertNotIn("secret", repr(raw))
            memory.close()

    def test_unapproved_or_unresolved_outcome_cannot_be_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            for result in ({"status": "clarification", "verdict": {"approved": True}},
                           {"status": "solution", "verdict": {"approved": False}}):
                with self.assertRaises(ValueError):
                    memory.confirm(result)
            self.assertEqual(memory.lookup("upload", "test.pdf")["confirmed_episodes"], [])
            memory.close()

    def test_access_episode_records_only_synthetic_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.confirm({"status": "solution", "category": "access",
                            "message": "private account", "course_code": "CS202",
                            "verdict": {"approved": True},
                            "evidence": [{"id": "check_course_access", "data": {"enrolled": False}}]})
            episodes = memory.lookup("access")["confirmed_episodes"]
            self.assertEqual(episodes[0]["resolution_kind"], "request_enrollment")
            self.assertNotIn("private account", repr(memory.connection.execute("SELECT * FROM episodes").fetchall()))
            memory.close()

    def test_login_confirmation_stores_no_demo_code_or_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.confirm({"status": "solution", "category": "login", "demo_account_code": "DEMO_LOCKED",
                            "message": "private message", "verdict": {"approved": True},
                            "evidence": [{"id": "check_demo_account", "data": {"status": "locked"}}]})
            self.assertEqual(memory.lookup("login")["confirmed_episodes"][0]["issue_code"],
                             "demo_account_locked")
            stored = repr(memory.connection.execute("SELECT * FROM episodes").fetchall())
            self.assertNotIn("DEMO_LOCKED", stored)
            self.assertNotIn("private message", stored)
            memory.close()

    def test_new_demo_causes_are_recorded_only_after_confirmation(self):
        cases = [("access", "check_course_access", {"course_open": False}, "course_not_open"),
                 ("login", "check_demo_account", {"login_service": "unavailable"}, "demo_login_unavailable"),
                 ("materials", "check_material_status", {"storage_status": "unavailable"},
                  "material_storage_unavailable"),
                 ("grades", "check_grade_status", {"status": "unpublished"}, "grade_unpublished")]
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            for category, tool, observation, expected in cases:
                result = {"status": "solution", "category": category, "verdict": {"approved": True},
                          "evidence": [{"id": tool, "data": observation}]}
                self.assertEqual(memory.confirm(result)["issue_code"], expected)
            memory.close()

    def test_knowledge_version_and_expiration(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            procedure = base / "upload.md"
            procedure.write_text("v1", encoding="utf-8")
            memory = MemoryStore(base / "memory.sqlite3")
            version = memory.sync_local_procedure("upload", procedure)
            self.assertEqual(memory.sync_local_procedure("upload", procedure), version)
            self.assertEqual(memory.lookup("upload")["knowledge"][0]["version"], version)
            memory.connection.execute("UPDATE knowledge SET expires_at = 0")
            memory.connection.commit()
            self.assertEqual(memory.lookup("upload")["knowledge"], [])
            memory.sync_local_procedure("upload", procedure)
            self.assertEqual(memory.lookup("upload")["knowledge"], [])
            procedure.write_text("v2", encoding="utf-8")
            self.assertNotEqual(memory.sync_local_procedure("upload", procedure), version)
            self.assertEqual(len(memory.lookup("upload")["knowledge"]), 1)
            memory.close()

    def test_expired_procedure_escalates_before_model_diagnosis(self):
        class Client:
            def ask(self, role, schema, data):
                if role == "dispatcher":
                    return {"category": "upload", "reason": "test"}
                raise AssertionError("Устаревшую инструкцию нельзя применять")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            procedure = root / "upload.md"
            procedure.write_text("v1", encoding="utf-8")
            memory = MemoryStore(root / "memory.sqlite3")
            memory.sync_local_procedure("upload", procedure)
            memory.connection.execute("UPDATE knowledge SET expires_at = 0")
            memory.connection.commit()
            journal = Journal(root / "turn")
            graph = build_graph(Client(), journal, {"max_tool_calls": 3, "max_revisions": 2}, memory)
            result = run_turn(graph, journal, "Не загружается файл", "lab.pdf", 28, [])
            self.assertEqual(result["status"], "escalation")
            self.assertIn("knowledge_expired", (root / "turn" / "events.jsonl").read_text())
            memory.close()

    def test_graph_reads_confirmed_memory_but_evidence_stays_fresh(self):
        class Client:
            def ask(self, role, schema, data):
                if role == "dispatcher":
                    return {"category": "upload", "reason": "test"}
                if role == "diagnostician":
                    assert data["memory_context"]["confirmed_episodes"]
                    return {"action": "check_file" if not data["evidence"] else "finish", "reason": "test"}
                if role == "resolver":
                    assert data["memory_context"]["confirmed_episodes"]
                    return {"mode": "clarification", "answer": "Какой текст ошибки?", "evidence_ids": ["check_file"]}
                return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.confirm({"status": "solution", "category": "upload", "filename": "old.pdf",
                            "verdict": {"approved": True},
                            "evidence": [{"id": "check_file", "data": {"errors": ["file_too_large"]}}]})
            journal = Journal(Path(tmp) / "turn")
            graph = build_graph(Client(), journal, {"max_tool_calls": 3, "max_revisions": 2}, memory)
            result = run_turn(graph, journal, "Не работает загрузка", "new.pdf", 10, [])
            self.assertEqual(result["status"], "clarification")
            self.assertEqual(result["evidence"][0]["data"]["errors"], [])
            memory.close()


if __name__ == "__main__":
    unittest.main()
