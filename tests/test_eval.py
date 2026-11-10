import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

from evaluation import eval_suite
from evaluation.eval_suite import CASES, MemorySampler, automatic_checks, summarize_events, load_cases, source_fingerprint
from evaluation.eval_summary import summarize, save_summary, percentile_nearest_rank


class EvalTests(unittest.TestCase):
    def test_catches_the_false_format_advice_from_real_run(self):
        result = {"status": "solution", "verdict": {"approved": True},
                  "final": "Файл 28 МБ превышает лимит 20 МБ. Сожмите PDF или отправьте в другом формате."}
        checks = automatic_checks(CASES[0], result)
        self.assertTrue(checks["mode_matches"])
        self.assertFalse(checks["no_other_format"])

    def test_event_summary_counts_retries(self):
        rows = [{"event": "llm_result", "role": "resolver", "wall_seconds": 3},
                {"event": "validation", "approved": False},
                {"event": "judge_disagreement", "judge_issues": ["false positive"]},
                {"event": "llm_result", "role": "resolver", "wall_seconds": 4},
                {"event": "validation", "approved": True}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            summary = summarize_events(path)
            self.assertEqual(summary["llm_calls"], 2)
            self.assertEqual(summary["by_role_seconds"]["resolver"], 7)
            self.assertEqual(summary["validation_attempts"], 2)
            self.assertEqual(summary["judge_disagreements"], 1)

    def test_service_available_flags_unconfirmed_outage(self):
        case = next(item for item in CASES if item["id"] == "service_available")
        result = {"status": "clarification", "verdict": {"approved": True},
                  "final": "Учебный сайт недоступен. Уточните, какой сервис не работает?"}
        self.assertFalse(automatic_checks(case, result)["no_unconfirmed_outage"])

    def test_login_checks_reject_password_request_and_reasking_known_code(self):
        active = next(item for item in CASES if item["id"] == "login_active")
        answer = {"status": "clarification", "verdict": {"approved": True},
                  "final": "Укажите код тестового профиля и сообщите пароль."}
        checks = automatic_checks(active, answer)
        self.assertFalse(checks["no_password_request"])
        self.assertFalse(checks["does_not_reask_demo_code"])

    def test_login_checks_catch_false_claims_in_actual_model_answers(self):
        observations = {
            "login_active": ("Тестовый профиль симулятора активен (DEMO_OK), но вход заблокирован. "
                             "Уточните точный текст ошибки или этап входа.", "no_unverified_block"),
            "login_missing": ("Тестовый профиль симулятора заблокирован. Уточните код DEMO_OK или DEMO_LOCKED.",
                              "no_unverified_account_state"),
        }
        for case_id, (answer, check) in observations.items():
            with self.subTest(case=case_id):
                case = next(item for item in CASES if item["id"] == case_id)
                result = {"status": "clarification", "verdict": {"approved": True}, "final": answer}
                self.assertFalse(automatic_checks(case, result)[check])

    def test_login_checks_catch_course_hallucination_and_allow_error_code_question(self):
        missing = next(item for item in CASES if item["id"] == "login_missing")
        active = next(item for item in CASES if item["id"] == "login_active")
        wrong = {"status": "clarification", "verdict": {"approved": True},
                 "final": "Профиль не зачислен на курс. Уточните код тестового профиля."}
        good = {"status": "clarification", "verdict": {"approved": True},
                "final": "Тестовый профиль активен. Уточните код ошибки при входе."}
        self.assertFalse(automatic_checks(missing, wrong)["no_unverified_course_claim"])
        self.assertTrue(automatic_checks(active, good)["does_not_reask_demo_code"])

    def test_sampler_is_optional_outside_windows(self):
        with MemorySampler() as memory:
            pass
        self.assertIsInstance(memory.samples, list)

    def test_development_and_acceptance_sets_are_distinct_and_balanced(self):
        dev = load_cases("dev")
        acceptance = load_cases("acceptance")
        self.assertEqual((len(dev), len(acceptance)), (23, 30))
        self.assertFalse({x["id"] for x in dev} & {x["id"] for x in acceptance})
        self.assertEqual(len(source_fingerprint()), 64)

    def test_out_of_scope_local_escalation_is_not_mislabeled_validator_approval(self):
        case = next(x for x in load_cases("acceptance") if x["id"] == "a26_billing")
        checks = automatic_checks(case, {"status": "escalation", "final": "Создана локальная учебная заявка."})
        self.assertTrue(all(checks.values()))
        self.assertNotIn("validator_approved", checks)

    def test_acceptance_cases_receive_factual_checks_independent_of_case_id(self):
        cases = {x["id"]: x for x in load_cases("acceptance")}
        bad_pdf = {"status": "solution", "verdict": {"approved": True},
                   "final": "Отправьте файл в другом формате."}
        bad_grade = {"status": "clarification", "verdict": {"approved": True},
                     "final": "Оценка ещё проверяется. Какой текст ошибки?"}
        self.assertFalse(automatic_checks(cases["a02_pdf_28"], bad_pdf)["mentions_pdf_limit"])
        self.assertFalse(automatic_checks(cases["a02_pdf_28"], bad_pdf)["no_disallowed_format"])
        self.assertFalse(automatic_checks(cases["a25_grade_published"], bad_grade)["no_false_unpublished"])

    def test_batch_resume_skips_completed_case_on_the_same_fingerprint(self):
        calls = []

        class FakeOllama:
            def __init__(self, config, journal):
                self.journal = journal

            def prepare(self):
                pass

            def close(self):
                pass

        def fake_run(*args, **kwargs):
            calls.append(1)
            return {"status": "solution", "final": "Файл превышает 20 МБ. Сожмите PDF.",
                    "verdict": {"approved": True}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"main_model":"fake"}', encoding="utf-8")
            patches = (patch.object(eval_suite, "ROOT", root),
                       patch.object(eval_suite, "source_fingerprint", return_value="fixed"),
                       patch.object(eval_suite, "Ollama", FakeOllama),
                       patch.object(eval_suite, "build_graph", return_value=object()),
                       patch.object(eval_suite, "run_turn", side_effect=fake_run),
                       patch.object(eval_suite, "summarize_events", return_value={"llm_calls": 0}))
            for item in patches:
                item.start()
            try:
                with patch.object(sys, "argv", ["eval_suite.py", "--case", "oversize_pdf"]), redirect_stdout(io.StringIO()):
                    eval_suite.main()
                output = next((root / "runtime").glob("eval_*/eval_results.json"))
                with patch.object(sys, "argv", ["eval_suite.py", "--case", "oversize_pdf",
                                                "--resume", str(output)]), redirect_stdout(io.StringIO()):
                    eval_suite.main()
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["planned_count"], 1)
            finally:
                for item in reversed(patches):
                    item.stop()

    def test_summary_keeps_failed_checks_and_alerts_visible(self):
        report = {"started_at": "synthetic", "cases": [
            {"case": {"id": "good", "expected": "solution"}, "status": "solution",
             "checks": {"mode_matches": True, "validator_approved": True},
             "wall_seconds": 100, "peak_system_ram_gb": 12.0,
             "llm_calls": 5, "validation_attempts": 1, "manual_grade": None},
            {"case": {"id": "bad", "expected": "clarification"}, "status": "escalation",
             "checks": {"mode_matches": False, "validator_approved": False},
             "wall_seconds": 227, "peak_system_ram_gb": 14.2,
             "llm_calls": 10, "validation_attempts": 3, "manual_grade": False},
        ]}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "eval_results.json"
            source.write_text(json.dumps(report), encoding="utf-8")
            summary = save_summary(source)
            self.assertEqual(summary["mode_matches"], 1)
            self.assertEqual(summary["all_automatic_checks_pass"], 1)
            self.assertEqual(summary["validation_retries"], 2)
            self.assertIsNone(summary["prompt_tokens"])
            self.assertIn("входные нет данных", (Path(tmp) / "eval_summary.md").read_text(encoding="utf-8"))
            self.assertEqual(summary["latency_p95_seconds"], 227)
            alerts = (Path(tmp) / "alerts.jsonl").read_text(encoding="utf-8")
            for kind in ("automatic_check_failed", "slow_case", "high_system_ram", "validator_retry_budget"):
                self.assertIn(kind, alerts)
            self.assertIn("bad", (Path(tmp) / "eval_summary.md").read_text(encoding="utf-8"))

    def test_incomplete_run_never_counts_as_success(self):
        summary, alerts = summarize({"cases": [{"case": {"id": "failed"}, "error": "HTTP 500"}]})
        self.assertEqual(summary["cases_completed"], 0)
        self.assertEqual(summary["mode_matches"], 0)
        self.assertEqual(percentile_nearest_rank([], .95), None)
        self.assertEqual(alerts[0]["kind"], "execution_error")


if __name__ == "__main__":
    unittest.main()
