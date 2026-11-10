import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from support_agent.workflow import build_graph, run_turn
from support_agent.tools import execute_tool
from support_agent.telemetry import Journal
from support_agent.llm import Ollama
from support_agent.models import (SolutionDraft, decision_schema, ClarificationReview, Route, ClarificationDraft, Verdict)
from support_agent.policy import (known_file_problem, enforce_verdict, bind_evidence, clarification_verdict, clarification_task, required_resolution_mode, login_clarification_fallback, catalog_clarification_fallback, grounded_solution_fallback,
                                  adjudicate_grounded_fallback)
from observability.trace_view import load_trace, summarize_turn


class FakeClient:
    def __init__(self, reject_count=0, category="upload"):
        self.reject_count, self.category = reject_count, category
        self.calls = []
        self.validations = 0

    def ask(self, role, schema, data):
        self.calls.append(role)
        if role == "dispatcher":
            result = {"category": self.category, "reason": "test"}
        elif role == "diagnostician":
            action = "check_file" if not data["evidence"] else "finish"
            result = {"action": action, "reason": "test"}
        elif role == "resolver":
            result = {"mode": "solution", "answer": "Уменьшите PDF до 20 МБ.", "evidence_ids": ["check_file"]}
        else:
            self.validations += 1
            approved = self.validations > self.reject_count
            result = {"approved": approved, "issues": [] if approved else ["Исправить ответ"]}
        return schema.model_validate(result).model_dump()


class FlowTests(unittest.TestCase):
    def run_flow(self, client, directory):
        return build_graph(client, Journal(directory), {"max_tool_calls": 3, "max_revisions": 2}).invoke(
            {"message": "Не загружается PDF", "filename": "lab.pdf", "size_mb": 28,
             "evidence": [], "tool_calls": 0, "revisions": 0}, {"recursion_limit": 35})

    def test_approved(self):
        with tempfile.TemporaryDirectory() as t:
            c = FakeClient()
            s = self.run_flow(c, t)
            self.assertEqual(s["status"], "solution")
            self.assertEqual(s["evidence"][0]["data"]["errors"], ["file_too_large"])
            self.assertEqual(c.calls, ["dispatcher", "diagnostician", "diagnostician", "resolver", "validator"])

    def test_rejected_then_fixed(self):
        with tempfile.TemporaryDirectory() as t:
            c = FakeClient(reject_count=1)
            s = self.run_flow(c, t)
            self.assertEqual(s["status"], "solution")
            self.assertEqual(c.validations, 2)

    def test_retry_budget(self):
        with tempfile.TemporaryDirectory() as t:
            c = FakeClient(reject_count=100)
            s = self.run_flow(c, t)
            self.assertEqual(s["status"], "escalation")
            self.assertEqual(c.validations, 3)
            self.assertTrue((Path(t) / "ticket.json").exists())

    def test_judge_disagreements_on_grounded_templates_recover_batch(self):
        scenarios = [
            ("upload", "unsupported_format", "", "", "", "", "lab.docx", 3, "solution"),
            ("access", "course_not_enrolled", "CS202", "", "", "", "", None, "solution"),
            ("access", "course_not_open", "CS303", "", "", "", "", None, "solution"),
            ("login", "login_locked", "", "DEMO_LOCKED", "", "", "", None, "solution"),
            ("login", "login_service_down", "", "DEMO_OFFLINE", "", "", "", None, "solution"),
            ("materials", "material_draft", "", "", "MAT102", "", "", None, "solution"),
            ("materials", "material_storage_down", "", "", "MAT103", "", "", None, "solution"),
            ("grades", "grade_unpublished", "", "", "", "SUB102", "", None, "solution"),
            ("materials", "material_published", "", "", "MAT101", "", "", None, "clarification"),
        ]

        class ErraticJudge:
            def __init__(self, category):
                self.category = category
                self.reviews = 0

            def ask(self, role, schema, data):
                if role == "dispatcher":
                    return {"category": self.category, "reason": "test"}
                if role == "diagnostician":
                    return {"action": data["allowed_actions"][0], "reason": "test"}
                if role == "resolver":
                    expected = data["required_mode"]
                    return {"mode": expected, "answer": "Недостаточно конкретный ответ.",
                            "evidence_ids": [item["id"] for item in data["evidence"]]}
                self.reviews += 1
                if schema is ClarificationReview:
                    return {"requests_missing_information": False,
                            "unsupported_claims": ["Ложное замечание"], "inappropriate_requests": []}
                return {"approved": False, "issues": ["Ложное замечание"]}

        for category, case, course, login, material, submission, filename, size, expected in scenarios:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                client = ErraticJudge(category)
                journal = Journal(tmp)
                result = run_turn(build_graph(client, journal,
                                  {"max_tool_calls": 3, "max_revisions": 2}), journal,
                                  "Не работает", filename, size, [], course, login, material, submission)
                self.assertEqual(result["status"], expected)
                self.assertEqual(client.reviews, 2)
                self.assertIn("judge_disagreement", (Path(tmp) / "events.jsonl").read_text(encoding="utf-8"))

    def test_adjudication_never_accepts_modified_or_unsupported_draft(self):
        state = {"category": "grades", "evidence": [execute_tool("check_grade_status",
                 {"submission_code": "SUB102"})], "fallback_kind": "solution"}
        state["draft"] = grounded_solution_fallback(state)
        self.assertIsNotNone(adjudicate_grounded_fallback(state,
                                 {"approved": False, "issues": ["ошибка валидатора"]}))
        state["draft"] = {**state["draft"], "answer": "Работа проверена и опубликована."}
        self.assertIsNone(adjudicate_grounded_fallback(state,
                              {"approved": False, "issues": ["ошибка валидатора"]}))
        state["draft"] = grounded_solution_fallback(state)
        state["evidence"][0]["data"]["result_available"] = False
        self.assertIsNone(adjudicate_grounded_fallback(state,
                              {"approved": False, "issues": ["ошибка валидатора"]}))

    def test_other_category(self):
        with tempfile.TemporaryDirectory() as t:
            c = FakeClient(category="other")
            s = self.run_flow(c, t)
            self.assertEqual(s["status"], "escalation")
            self.assertEqual(c.calls, ["dispatcher"])

    def test_service_status_uses_own_tool_and_never_claims_real_platform(self):
        class ServiceClient(FakeClient):
            def __init__(self, expected):
                super().__init__(category="service")
                self.expected = expected

            def ask(self, role, schema, data):
                if role == "diagnostician":
                    self.assert_actions(data)
                    return {"action": "finish" if data["evidence"] else "get_service_status", "reason": "проверка симулятора"}
                if role == "resolver":
                    if data["category"] != "service" or data["required_mode"] != self.expected:
                        raise AssertionError("Service policy was not applied")
                    answer = ("Симулятор сообщает о недоступности сайта. Попробуйте позже."
                              if self.expected == "solution" else "Какой текст ошибки и адрес страницы вы видите?")
                    return {"mode": self.expected, "answer": answer,
                            "evidence_ids": ["get_service_status"]}
                if role == "validator" and schema is ClarificationReview:
                    return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
                return super().ask(role, schema, data)

            def assert_actions(self, data):
                if data["allowed_actions"] != (["finish"] if data["evidence"] else ["get_service_status"]):
                    raise AssertionError("Unrelated upload tool available to service agent")

        for simulator_status, expected in (("available", "clarification"), ("unavailable", "solution")):
            with self.subTest(simulator_status=simulator_status), tempfile.TemporaryDirectory() as t:
                client = ServiceClient(expected)
                result = build_graph(client, Journal(t), {"max_tool_calls": 3, "max_revisions": 2,
                                                         "simulator_service_status": simulator_status}).invoke(
                    {"message": "Учебный сайт не открывается", "evidence": [], "tool_calls": 0,
                     "revisions": 0}, {"recursion_limit": 35})
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["evidence"][0]["data"]["platform_service"], simulator_status)
                self.assertEqual(result["tool_calls"], 1)

    def test_access_route_distinguishes_demo_enrollment_and_missing_code(self):
        class AccessClient(FakeClient):
            def __init__(self, expected, code):
                super().__init__(category="access")
                self.expected, self.code = expected, code

            def ask(self, role, schema, data):
                if role == "diagnostician":
                    if data["allowed_actions"] != (["finish"] if data["evidence"] else ["check_course_access"]):
                        raise AssertionError("Access route exposed unrelated tools")
                    return {"action": "finish" if data["evidence"] else "check_course_access", "reason": "test"}
                if role == "resolver":
                    if data["required_mode"] != self.expected:
                        raise AssertionError("Wrong access resolution policy")
                    return {"mode": self.expected,
                            "answer": "Тестовый профиль не зачислен на курс. Обратитесь к преподавателю."
                            if self.expected == "solution" else
                            "Какой текст ошибки вы видите?" if self.code == "CS101" else
                            "Уточните код курса или текст ошибки.",
                            "evidence_ids": ["check_course_access"]}
                if role == "validator" and schema is ClarificationReview:
                    return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
                return super().ask(role, schema, data)

        for code, expected in (("", "clarification"), ("CS101", "clarification"),
                               ("CS202", "solution"), ("UNKNOWN", "clarification")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as t:
                journal = Journal(t)
                state = run_turn(build_graph(AccessClient(expected, code), journal,
                                             {"max_tool_calls": 3, "max_revisions": 2}),
                                 journal, "Не вижу курс", "", None, [], code)
                self.assertEqual(state["status"], expected)
                self.assertEqual(state["tool_calls"], 1)
                self.assertEqual(state["evidence"][0]["data"]["course_code"], code)
                self.assertFalse(state["evidence"][0]["data"]["actual_account_read"])

    def test_access_solution_must_name_simulator_even_if_validator_approves(self):
        state = {"category": "access", "evidence": [execute_tool("check_course_access", {"course_code": "CS202"})],
                 "draft": {"mode": "solution", "answer": "Вы не зачислены на курс CS202.",
                           "evidence_ids": ["check_course_access"]}}
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        state["draft"]["answer"] = "Тестовый профиль симулятора не зачислен на CS202. Обратитесь к преподавателю."
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_login_route_uses_only_demo_account_and_modes_depend_on_observation(self):
        class LoginClient(FakeClient):
            def __init__(self, expected, code):
                super().__init__(category="login")
                self.expected, self.code = expected, code

            def ask(self, role, schema, data):
                if role == "diagnostician":
                    allowed = ["finish"] if data["evidence"] else ["check_demo_account"]
                    self_outer.assertEqual(data["allowed_actions"], allowed)
                    return {"action": allowed[0], "reason": "test"}
                if role == "resolver":
                    self_outer.assertEqual(data["required_mode"], self.expected)
                    self_outer.assertEqual(data["demo_account_code"], self.code)
                    return {"mode": self.expected,
                            "answer": ("Тестовый профиль симулятора заблокирован. Обратитесь к администратору."
                                       if self.expected == "solution" else
                                       "Какой текст ошибки появляется при входе?" if self.code == "DEMO_OK" else
                                       "Уточните код тестового профиля DEMO_OK или DEMO_LOCKED."),
                            "evidence_ids": ["check_demo_account"]}
                if role == "validator" and schema is ClarificationReview:
                    return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
                return super().ask(role, schema, data)

        self_outer = self
        for code, expected in (("", "clarification"), ("DEMO_OK", "clarification"),
                               ("DEMO_LOCKED", "solution"), ("NOT_IN_DEMO", "clarification")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as t:
                journal = Journal(t)
                result = run_turn(build_graph(LoginClient(expected, code), journal,
                                              {"max_tool_calls": 3, "max_revisions": 2}), journal,
                                  "Не могу войти", "", None, [], demo_account_code=code)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["tool_calls"], 1)
                self.assertEqual(result["evidence"][0]["id"], "check_demo_account")
                self.assertFalse(result["evidence"][0]["data"]["actual_account_read"])

    def test_login_guard_rejects_real_account_claim_or_password_request(self):
        state = {"category": "login", "evidence": [execute_tool("check_demo_account",
                 {"demo_account_code": "DEMO_LOCKED"})],
                 "draft": {"mode": "solution", "answer": "Ваш реальный аккаунт заблокирован. Сообщите пароль.",
                           "evidence_ids": ["check_demo_account"]}}
        verdict = enforce_verdict(state, {"approved": True, "issues": []})
        self.assertFalse(verdict["approved"])
        self.assertTrue(any("пароль" in issue for issue in verdict["issues"]))
        state["draft"]["answer"] = "Тестовый профиль симулятора заблокирован. Обратитесь к администратору."
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        state["draft"]["mode"] = "clarification"
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_active_demo_account_does_not_justify_solved_claim_or_reask_code(self):
        state = {"category": "login", "evidence": [execute_tool("check_demo_account",
                 {"demo_account_code": "DEMO_OK"})],
                 "draft": {"mode": "solution", "answer": "Ваш вход работает.",
                           "evidence_ids": ["check_demo_account"]}}
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        state["draft"].update(mode="clarification", answer="Уточните код тестового профиля.")
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        state["draft"]["answer"] = "Какой точный текст ошибки появляется при входе?"
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_login_clarification_blocks_false_claims_from_observed_runs(self):
        examples = [
            ("DEMO_OK", "Тестовый профиль симулятора активен (DEMO_OK), но вход заблокирован. "
                        "Уточните точный текст ошибки или этап входа, на котором возникает проблема."),
            ("", "Тестовый профиль симулятора заблокирован. Уточните код DEMO_OK или DEMO_LOCKED."),
        ]
        for code, answer in examples:
            with self.subTest(code=code):
                state = {"category": "login", "evidence": [execute_tool("check_demo_account",
                         {"demo_account_code": code})],
                         "draft": {"mode": "clarification", "answer": answer,
                                   "evidence_ids": ["check_demo_account"]}}
                verdict = enforce_verdict(state, {"approved": True, "issues": []})
                self.assertFalse(verdict["approved"])
                self.assertIn("Блокировка не подтверждена", " ".join(verdict["issues"]))
                state["draft"]["answer"] = ("Тестовый профиль симулятора активен. Какой текст ошибки возникает?"
                                              if code else "Уточните код тестового профиля DEMO_OK или DEMO_LOCKED.")
                self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_login_unknown_code_cannot_invent_course_enrollment(self):
        state = {"category": "login", "evidence": [execute_tool("check_demo_account", {})],
                 "draft": {"mode": "clarification", "evidence_ids": ["check_demo_account"],
                           "answer": "Тестовый профиль симулятора не зачислен на указанный курс. "
                                     "Уточните код тестового профиля или текст ошибки."}}
        verdict = enforce_verdict(state, {"approved": True, "issues": []})
        self.assertFalse(verdict["approved"])
        self.assertIn("Зачисление на курс", " ".join(verdict["issues"]))

    def test_login_error_code_question_is_allowed_for_active_profile(self):
        state = {"category": "login", "evidence": [execute_tool("check_demo_account",
                 {"demo_account_code": "DEMO_OK"})],
                 "draft": {"mode": "clarification", "evidence_ids": ["check_demo_account"],
                           "answer": "Тестовый профиль активен. Уточните код ошибки при входе."}}
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_repeated_rejected_login_answer_recovers_with_reviewed_question(self):
        class RepeatingLoginClient:
            def ask(self, role, schema, data):
                if role == "dispatcher":
                    return {"category": "login", "reason": "test"}
                if role == "diagnostician":
                    return {"action": "finish" if data["evidence"] else "check_demo_account",
                            "reason": "test"}
                if role == "resolver":
                    return {"mode": "clarification", "answer":
                            "Тестовый профиль симулятора активен, но вход заблокирован. Уточните текст ошибки.",
                            "evidence_ids": ["check_demo_account"]}
                if role == "validator" and schema is ClarificationReview:
                    blocked = "заблокирован" in data["candidate_answer"]
                    return {"requests_missing_information": True,
                            "unsupported_claims": ["Блокировка не подтверждена"] if blocked else [],
                            "inappropriate_requests": []}
                raise AssertionError(role)

        with tempfile.TemporaryDirectory() as tmp:
            journal = Journal(tmp)
            graph = build_graph(RepeatingLoginClient(), journal,
                                {"max_tool_calls": 3, "max_revisions": 2})
            result = run_turn(graph, journal, "Не могу войти", "", None, [],
                              demo_account_code="DEMO_OK")
            self.assertEqual(result["status"], "clarification")
            self.assertEqual(result["revisions"], 2)
            self.assertIn("код ошибки", result["final"])
            self.assertNotIn("заблокирован", result["final"])
            self.assertFalse((Path(tmp) / "ticket.json").exists())
            self.assertIn("clarification_fallback", (Path(tmp) / "events.jsonl").read_text(encoding="utf-8"))
            _, turns = load_trace(Path(tmp) / "events.jsonl")
            trajectory = summarize_turn(*next(iter(turns.items())))["trajectory"]
            self.assertIn("уточнение: резервный вопрос по результату инструмента", trajectory)

    def test_login_fallback_never_invents_unknown_account_state(self):
        state = {"evidence": [execute_tool("check_demo_account", {})]}
        draft = login_clarification_fallback(state)
        self.assertEqual(draft["evidence_ids"], ["check_demo_account"])
        self.assertNotIn("заблокирован", draft["answer"])
        self.assertNotIn("зачислен", draft["answer"])

    def test_material_and_grade_routes_use_only_their_simulated_observations(self):
        class CatalogClient:
            def __init__(self, category, expected):
                self.category, self.expected = category, expected

            def ask(self, role, schema, data):
                tool = "check_material_status" if self.category == "materials" else "check_grade_status"
                if role == "dispatcher":
                    return {"category": self.category, "reason": "test"}
                if role == "diagnostician":
                    expected_actions = ["finish"] if data["evidence"] else [tool]
                    self_outer.assertEqual(data["allowed_actions"], expected_actions)
                    return {"action": expected_actions[0], "reason": "test"}
                if role == "resolver":
                    self_outer.assertEqual(data["required_mode"], self.expected)
                    if self.expected == "solution":
                        return {"mode": "solution", "answer":
                                "Тестовый материал не опубликован в симуляторе. Уточните у преподавателя."
                                if self.category == "materials" else
                                "Тестовая работа ещё проверяется в симуляторе. Дождитесь результата.",
                                "evidence_ids": [tool]}
                    return {"mode": "clarification", "answer":
                            "Какой текст ошибки появляется при просмотре?" if data.get("material_code") == "MAT101"
                            or data.get("submission_code") == "SUB103" else
                            "Уточните код тестового материала или сдачи.", "evidence_ids": [tool]}
                if role == "validator" and schema is ClarificationReview:
                    return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
                return {"approved": True, "issues": []}

        self_outer = self
        examples = [("materials", "MAT102", "", "solution"),
                    ("materials", "MAT101", "", "clarification"),
                    ("materials", "", "", "clarification"),
                    ("grades", "", "SUB101", "solution"),
                    ("grades", "", "SUB102", "solution"),
                    ("grades", "", "SUB103", "clarification"),
                    ("grades", "", "", "clarification")]
        for category, material, submission, expected in examples:
            with self.subTest(category=category, material=material, submission=submission), tempfile.TemporaryDirectory() as tmp:
                journal = Journal(tmp)
                graph = build_graph(CatalogClient(category, expected), journal,
                                    {"max_tool_calls": 3, "max_revisions": 2})
                result = run_turn(graph, journal, "Не вижу страницу", "", None, [],
                                  material_code=material, submission_code=submission)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["tool_calls"], 1)
                self.assertFalse(result["evidence"][0]["data"]["actual_platform_read"])

    def test_material_and_grade_false_states_are_rejected(self):
        examples = [("materials", "check_material_status", "material_code", "MAT101",
                     "Материал не опубликован. Какой текст ошибки?"),
                    ("grades", "check_grade_status", "submission_code", "SUB103",
                     "Оценка ещё проверяется. Какой текст ошибки?")]
        for category, tool, field, code, answer in examples:
            with self.subTest(category=category):
                state = {"category": category, "evidence": [execute_tool(tool, {field: code})],
                         "draft": {"mode": "clarification", "answer": answer, "evidence_ids": [tool]}}
                self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
                state["draft"]["answer"] = "Какой точный текст ошибки возникает при просмотре?"
                self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_additional_demo_causes_require_solution_without_fake_real_access(self):
        examples = [("access", "check_course_access", {"course_code": "CS303"}, "course_open"),
                    ("login", "check_demo_account", {"demo_account_code": "DEMO_OFFLINE"}, "login_service"),
                    ("materials", "check_material_status", {"material_code": "MAT103"}, "storage_status")]
        for category, tool, args, field in examples:
            with self.subTest(category=category):
                observation = execute_tool(tool, args)
                self.assertIn(observation["data"].get(field), (False, "unavailable"))
                state = {"category": category, "evidence": [observation]}
                self.assertEqual(required_resolution_mode(state), "solution")
                state["draft"] = {"mode": "clarification", "answer": "Какой текст ошибки?",
                                  "evidence_ids": [tool]}
                self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_catalog_fallback_uses_only_the_observed_status(self):
        for category, tool, field, code in (("materials", "check_material_status", "material_code", "MAT101"),
                                           ("grades", "check_grade_status", "submission_code", "SUB103")):
            with self.subTest(category=category):
                state = {"category": category, "evidence": [execute_tool(tool, {field: code})]}
                draft = catalog_clarification_fallback(state)
                state["draft"] = draft
                self.assertIn("текст ошибки", draft["answer"])
                self.assertNotIn("доступен", draft["answer"])
                self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
                state["evidence"] = [execute_tool(tool, {})]
                draft = catalog_clarification_fallback(state)
                self.assertNotIn("опубликован", draft["answer"])

    def test_actual_dev_run_false_answers_are_not_approved_again(self):
        cases = [
            ("upload", [execute_tool("get_upload_rules", {}), execute_tool("check_file", {"filename": "lab.pdf", "size_mb": 28})],
             "solution", "Файл превысил лимит. Предложите сжать PDF до 20 МБ."),
            ("upload", [execute_tool("get_upload_rules", {}), execute_tool("check_file", {"filename": "lab.docx", "size_mb": 3})],
             "solution", "Предложите сжать документ до формата PDF."),
            ("upload", [execute_tool("get_upload_rules", {}), execute_tool("check_file", {"filename": "lab.pdf", "size_mb": 10})],
             "clarification", "Ошибка не связана с размером файла. Какой текст ошибки?"),
            ("login", [execute_tool("check_demo_account", {})],
             "clarification", "Код профиля не распознан. Уточните код."),
            ("login", [execute_tool("check_demo_account", {"demo_account_code": "DEMO_OK"})],
             "clarification", "Ошибка не связана с доступностью демо-аккаунта. Какой текст ошибки?"),
            ("login", [execute_tool("check_demo_account", {"demo_account_code": "DEMO_LOCKED"})],
             "solution", "Тестовый профиль симулятора заблокирован. Уточните текст ошибки."),
            ("materials", [execute_tool("check_material_status", {})],
             "clarification", "Код материала не распознан. Уточните код MAT101."),
            ("materials", [execute_tool("check_material_status", {"material_code": "MAT101"})],
             "clarification", "Материал опубликованный и доступный. Какой текст ошибки?"),
            ("grades", [execute_tool("check_grade_status", {"submission_code": "SUB102"})],
             "solution", "Оценка не опубликована в симуляторе. Ждите результата проверки."),
            ("grades", [execute_tool("check_grade_status", {})],
             "clarification", "Код сдачи отсутствует в системе. Уточните его."),
            ("login", [execute_tool("check_demo_account", {"demo_account_code": "DEMO_OK"})],
             "clarification", "Ошибка входа не связана с доступностью демо-аккаунта. Какой текст ошибки?"),
            ("login", [execute_tool("check_demo_account", {})],
             "clarification", "Не указан код профиля. Укажите точный текст ошибки."),
            ("login", [execute_tool("check_demo_account", {"demo_account_code": "DEMO_OK"})],
             "clarification", "Проверка демо-аккаунта активна и вход доступен. Уточните текст ошибки."),
            ("materials", [execute_tool("check_material_status", {})],
             "clarification", "Не найдено материалов с указанным кодом. Уточните код лекции."),
            ("grades", [execute_tool("check_grade_status", {"submission_code": "SUB102"})],
             "solution", "Оценка не опубликована в симуляторе. Её увидите после завершения проверки преподавателем."),
        ]
        for category, evidence, mode, answer in cases:
            with self.subTest(category=category, answer=answer):
                state = {"category": category, "evidence": evidence,
                         "draft": {"mode": mode, "answer": answer,
                                   "evidence_ids": [item["id"] for item in evidence]}}
                self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_grounded_repair_for_five_avoidable_escalations_is_reviewable(self):
        examples = [
            ("access", execute_tool("check_course_access", {"course_code": "CS202"})),
            ("access", execute_tool("check_course_access", {"course_code": "CS303"})),
            ("materials", execute_tool("check_material_status", {"material_code": "MAT102"})),
            ("materials", execute_tool("check_material_status", {"material_code": "MAT103"})),
            ("grades", execute_tool("check_grade_status", {"submission_code": "SUB101"})),
        ]
        for category, observation in examples:
            with self.subTest(category=category, observation=observation):
                state = {"category": category, "evidence": [observation]}
                self.assertEqual(required_resolution_mode(state), "solution")
                state["draft"] = grounded_solution_fallback(state)
                self.assertIsNotNone(state["draft"])
                self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_rejected_known_solution_uses_guarded_repair_then_validator(self):
        class BadThenReviewedClient:
            def __init__(self):
                self.resolver_calls = 0
                self.reviewed = []

            def ask(self, role, schema, data):
                if role == "dispatcher":
                    return {"category": "materials", "reason": "test"}
                if role == "diagnostician":
                    return {"action": "finish" if data["evidence"] else "check_material_status", "reason": "test"}
                if role == "resolver":
                    self.resolver_calls += 1
                    return {"mode": "solution", "answer": "Материал опубликован.",
                            "evidence_ids": ["check_material_status"]}
                if role == "validator":
                    self.reviewed.append(data["draft"]["answer"])
                    return {"approved": len(self.reviewed) > 1, "issues": ["Нет шага"] if len(self.reviewed) == 1 else []}
                raise AssertionError(role)

        with tempfile.TemporaryDirectory() as tmp:
            client = BadThenReviewedClient()
            journal = Journal(tmp)
            result = run_turn(build_graph(client, journal, {"max_tool_calls": 3, "max_revisions": 2}),
                              journal, "Не вижу материал", "", None, [], material_code="MAT102")
            self.assertEqual(result["status"], "solution")
            self.assertEqual(client.resolver_calls, 1)
            self.assertEqual(len(client.reviewed), 2)
            self.assertIn("преподавателя", result["final"])
            self.assertIn("solution_fallback", (Path(tmp) / "events.jsonl").read_text(encoding="utf-8"))

    def test_enrolled_course_is_not_requested_again(self):
        state = {"category": "access", "course_code": "CS101",
                 "evidence": [execute_tool("check_course_access", {"course_code": "CS101"})],
                 "draft": {"mode": "clarification",
                           "answer": "Уточните, какой именно курс и номер задания вызывают проблему.",
                           "evidence_ids": ["check_course_access"]}}
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        state["draft"]["answer"] = "Тестовый профиль зачислен на CS101. Какой текст ошибки и название задания?"
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_missing_metadata_not_success(self):
        result = execute_tool("check_file", {})["data"]
        self.assertEqual(len(result["missing"]), 2)
        self.assertFalse(result["actual_file_read"])

    def test_tool_allowlist(self):
        with self.assertRaises(ValueError):
            execute_tool("run_shell", {})

    def test_nonfinite_size(self):
        with self.assertRaises(ValueError):
            execute_tool("check_file", {"size_mb": float("nan")})

    def problem_state(self, size=28, filename="lab.pdf"):
        state = {"filename": filename, "size_mb": size}
        state["evidence"] = [execute_tool(n, state) for n in ("get_upload_rules", "check_file")]
        return state

    def test_known_error_policy(self):
        self.assertTrue(known_file_problem(self.problem_state()))
        self.assertTrue(known_file_problem(self.problem_state(10, "lab.exe")))
        self.assertFalse(known_file_problem(self.problem_state(10)))
        self.assertFalse(known_file_problem(self.problem_state(None)))
        self.assertFalse(known_file_problem({"message": "file_too_large"}))

    def test_observed_false_approval_is_blocked(self):
        state = self.problem_state()
        state["draft"] = {"mode": "escalation", "answer": "Уточните формат", "evidence_ids": ["get_upload_rules", "check_file"]}
        result = enforce_verdict(state, {"approved": True, "issues": []})
        self.assertFalse(result["approved"])
        self.assertTrue(result["issues"])

    def test_pdf_only_rules_block_alternative_format_even_when_validator_approves(self):
        state = self.problem_state()
        state["draft"] = {"mode": "solution", "answer": "Сожмите PDF или отправьте файл в другом формате.",
                          "evidence_ids": ["check_file", "get_upload_rules"]}
        result = enforce_verdict(state, {"approved": True, "issues": []})
        self.assertFalse(result["approved"])
        self.assertIn("другой формат", " ".join(result["issues"]))
        state["draft"]["answer"] = "Размер 28 МБ превышает 20 МБ. Сожмите PDF до 20 МБ и повторите загрузку."
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_available_service_rejects_claim_of_confirmed_outage(self):
        state = {"category": "service", "evidence": [{"id": "get_service_status", "data":
                 {"source": "simulator", "platform_service": "available"}}],
                 "draft": {"mode": "clarification", "answer":
                           "Учебный сайт недоступен. Уточните, какая часть не работает?",
                           "evidence_ids": ["get_service_status"]}}
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        state["draft"]["answer"] = "Вы сообщаете, что сайт не открывается. Какой текст ошибки вы видите?"
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_solution_schema_and_guard(self):
        state = self.problem_state()
        draft = {"mode": "solution", "answer": "Сожмите PDF до 20 МБ.", "evidence_ids": ["get_upload_rules", "check_file"]}
        SolutionDraft.model_validate(draft)
        with self.assertRaises(ValueError):
            SolutionDraft.model_validate({**draft, "mode": "escalation"})
        state["draft"] = draft
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": ["Ошибка ответа"]})["approved"])

    def test_unknown_problem_requires_initial_clarification(self):
        state = self.problem_state(10)
        state["draft"] = {"mode": "escalation", "answer": "Нужен разбор", "evidence_ids": ["check_file"]}
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_resolution_policy_distinguishes_three_cases(self):
        self.assertEqual(required_resolution_mode(self.problem_state(28)), "solution")
        self.assertEqual(required_resolution_mode(self.problem_state(10)), "clarification")
        self.assertEqual(required_resolution_mode(self.problem_state(None, "")), "clarification")
        self.assertIsNone(required_resolution_mode({}))
        state = self.problem_state(10)
        state["evidence"].append({"id": "get_service_status", "data": {"upload_service": "unavailable"}})
        self.assertIsNone(required_resolution_mode(state))

    def test_valid_metadata_do_not_license_invented_cause(self):
        state = self.problem_state(10)
        state["draft"] = {"mode": "solution", "answer": "Ошибка из-за скрытых ограничений", "evidence_ids": ["get_upload_rules"]}
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])
        with self.assertRaises(ValueError):
            ClarificationDraft.model_validate(state["draft"])

    def test_valid_file_unknown_error_flow_requests_symptoms(self):
        class UnknownErrorClient(FakeClient):
            def ask(self, role, schema, data):
                if role == "diagnostician":
                    used = {x["id"] for x in data["evidence"]}
                    action = "get_upload_rules" if not used else "check_file" if "check_file" not in used else "finish"
                    return schema.model_validate({"action": action, "reason": "test"}).model_dump()
                if role == "resolver":
                    if schema is not ClarificationDraft or data["required_mode"] != "clarification":
                        raise AssertionError("Must constrain clarification for unknown cause")
                    return schema.model_validate({"mode": "clarification", "answer": "Указанные размер и формат подходят. Какой точный текст ошибки появляется?", "evidence_ids": ["get_upload_rules", "check_file"]}).model_dump()
                if role == "validator":
                    if schema is not ClarificationReview or data["missing_information"]:
                        raise AssertionError("No missing metadata, but still a clarification review")
                    return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
                return super().ask(role, schema, data)
        with tempfile.TemporaryDirectory() as t:
            state = build_graph(UnknownErrorClient(), Journal(t), {"max_tool_calls": 3, "max_revisions": 2}).invoke(
                {"message": "После выбора файла появляется неизвестная ошибка", "filename": "lab.pdf", "size_mb": 10,
                 "evidence": [], "tool_calls": 0, "revisions": 0}, {"recursion_limit": 35})
            self.assertEqual(state["status"], "clarification")
            self.assertEqual(state["revisions"], 1)
            self.assertFalse((Path(t) / "ticket.json").exists())

    def test_conflicting_rules_do_not_force_solution(self):
        state = self.problem_state()
        state["evidence"][0]["data"]["max_size_mb"] = 50
        self.assertFalse(known_file_problem(state))

    def test_missing_reference_bound_with_provenance(self):
        state = self.problem_state()
        original = {"mode": "solution", "answer": "Сожмите PDF до 20 МБ.", "evidence_ids": ["get_upload_rules"]}
        draft, provenance = bind_evidence(state, original)
        self.assertEqual(original["evidence_ids"], ["get_upload_rules"])
        self.assertEqual(provenance["model_evidence_ids"], ["get_upload_rules"])
        self.assertEqual(set(draft["evidence_ids"]), {"check_file", "get_upload_rules"})
        state["draft"] = draft
        self.assertTrue(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_binding_does_not_hide_hallucinated_reference(self):
        state = self.problem_state()
        state["draft"], _ = bind_evidence(state, {"mode": "solution", "answer": "Ответ", "evidence_ids": ["invented"]})
        self.assertFalse(enforce_verdict(state, {"approved": True, "issues": []})["approved"])

    def test_binding_does_not_override_semantic_rejection(self):
        state = self.problem_state()
        state["draft"], _ = bind_evidence(state, {"mode": "solution", "answer": "Лимит 100 МБ", "evidence_ids": []})
        self.assertFalse(enforce_verdict(state, {"approved": False, "issues": ["Неверный лимит"]})["approved"])

    def test_no_binding_for_missing_or_conflicting_evidence(self):
        draft = {"mode": "clarification", "answer": "Размер?", "evidence_ids": []}
        for state in ({}, self.problem_state(None)):
            bound, provenance = bind_evidence(state, draft)
            self.assertEqual(bound["evidence_ids"], [])
            self.assertEqual(provenance["policy_evidence_ids"], [])

    def test_regression_observed_single_reference_does_not_escalate(self):
        class ObservedClient(FakeClient):
            def ask(self, role, schema, data):
                if role == "diagnostician":
                    used = {e["id"] for e in data["evidence"]}
                    action = "get_upload_rules" if not used else "check_file" if "check_file" not in used else "finish"
                    return {"action": action, "reason": "test"}
                if role == "resolver":
                    return {"mode": "solution", "answer": "Файл превысил лимит 20 МБ. Сожмите PDF и повторите загрузку.", "evidence_ids": ["get_upload_rules"]}
                return super().ask(role, schema, data)
        with tempfile.TemporaryDirectory() as t:
            client = ObservedClient()
            state = self.run_flow(client, t)
            self.assertEqual(state["status"], "solution")
            self.assertEqual(client.validations, 1)
            self.assertFalse((Path(t) / "ticket.json").exists())
            self.assertEqual(state["evidence_binding"]["model_evidence_ids"], ["get_upload_rules"])

    def test_step_schema_excludes_completed_tools(self):
        schema = decision_schema(["get_service_status", "finish"])
        self.assertEqual(schema.model_json_schema()["properties"]["action"]["enum"], ["get_service_status", "finish"])
        for repeated in ("check_file", "get_upload_rules"):
            with self.assertRaises(ValueError):
                schema.model_validate({"action": repeated, "reason": "Повтор"})
        self.assertEqual(schema.model_validate({"action": "finish", "reason": "Готово"}).action, "finish")

    def test_graph_sends_current_action_schema(self):
        class SchemaClient(FakeClient):
            def ask(self, role, schema, data):
                if role == "diagnostician":
                    available = data["allowed_actions"]
                    prop = schema.model_json_schema()["properties"]["action"]
                    self.asserted_actions = prop.get("enum", [prop.get("const")])
                    if self.asserted_actions != available:
                        raise AssertionError("JSON schema differs from allowed actions")
                    used = {e["id"] for e in data["evidence"]}
                    for done in used:
                        if done in self.asserted_actions:
                            raise AssertionError("Completed tool allowed again")
                    action = "get_upload_rules" if not used else "check_file" if "check_file" not in used else "finish"
                    return schema.model_validate({"action": action, "reason": "test"}).model_dump()
                return super().ask(role, schema, data)
        with tempfile.TemporaryDirectory() as t:
            state = self.run_flow(SchemaClient(), t)
            self.assertEqual(state["status"], "solution")
            self.assertEqual(state["tool_calls"], 2)

    def test_clarification_review_does_not_require_missing_values(self):
        state = self.problem_state(None, "")
        state.update(message="Не могу загрузить лабораторную работу",
                     draft={"mode": "clarification", "answer": "Укажите расширение и размер в мегабайтах.", "evidence_ids": []})
        task = clarification_task(state)
        self.assertEqual(task["missing_information"], ["расширение файла", "размер файла"])
        self.assertIn("Укажите", task["candidate_answer"])
        review = {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
        self.assertTrue(clarification_verdict(review)["approved"])

    def test_clarification_not_automatically_accepted(self):
        reviews = [
            {"requests_missing_information": False, "unsupported_claims": [], "inappropriate_requests": []},
            {"requests_missing_information": True, "unsupported_claims": ["Выдумано превышение лимита"], "inappropriate_requests": []},
            {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": ["Запрошен пароль"]},
        ]
        for review in reviews:
            self.assertFalse(clarification_verdict(review)["approved"])

    def test_missing_metadata_flow_uses_special_review(self):
        class ClarificationClient(FakeClient):
            def ask(self, role, schema, data):
                if role == "resolver":
                    return {"mode": "clarification", "answer": "Для корректной проверки необходимо указать расширение файла (например, .pdf) и его размер в мегабайтах.", "evidence_ids": ["check_file"]}
                if role == "validator":
                    if schema is not ClarificationReview:
                        raise AssertionError("Wrong validator schema")
                    if data["missing_information"] != ["расширение файла", "размер файла"]:
                        raise AssertionError("Missing fields not passed")
                    return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
                return super().ask(role, schema, data)
        with tempfile.TemporaryDirectory() as t:
            state = build_graph(ClarificationClient(), Journal(t), {"max_tool_calls": 3, "max_revisions": 2}).invoke(
                {"message": "Не могу загрузить работу", "filename": "", "size_mb": None,
                 "evidence": [], "tool_calls": 0, "revisions": 0}, {"recursion_limit": 35})
            self.assertEqual(state["status"], "clarification")
            self.assertEqual(state["revisions"], 1)
            self.assertFalse((Path(t)/"ticket.json").exists())

    def test_follow_up_keeps_context_and_new_metadata_without_old_evidence(self):
        class FollowUpClient(FakeClient):
            def ask(self, role, schema, data):
                if role == "dispatcher":
                    if data["message"] == "lab.pdf, 28 МБ":
                        self.assert_history(data)
                    return {"category": "upload", "reason": "продолжение обращения"}
                if role == "diagnostician":
                    used = {item["id"] for item in data["evidence"]}
                    action = "get_upload_rules" if not used else "check_file" if "check_file" not in used else "finish"
                    return {"action": action, "reason": "проверка"}
                if role == "resolver" and schema is ClarificationDraft:
                    return {"mode": "clarification", "answer": "Укажите имя файла и размер в МБ.", "evidence_ids": ["check_file"]}
                if role == "validator" and schema is ClarificationReview:
                    return {"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []}
                return super().ask(role, schema, data)

            def assert_history(self, data):
                if data["conversation"][-1]["content"] != "Укажите имя файла и размер в МБ.":
                    raise AssertionError("Previous assistant question was lost")

        with tempfile.TemporaryDirectory() as t:
            client = FollowUpClient()
            graph, journal = build_graph(client, Journal(t), {"max_tool_calls": 3, "max_revisions": 2}), Journal(t)
            first = run_turn(graph, journal, "Не могу загрузить работу", "", None, [])
            self.assertEqual(first["status"], "clarification")
            history = [{"role": "user", "content": "Не могу загрузить работу"},
                       {"role": "assistant", "content": first["final"]}]
            second = run_turn(graph, journal, "lab.pdf, 28 МБ", "lab.pdf", 28, history)
            self.assertEqual(second["status"], "solution")
            self.assertEqual(second["evidence"][1]["data"]["errors"], ["file_too_large"])
            self.assertEqual(second["tool_calls"], 2)

    def test_ollama_selects_clarification_prompt_and_family(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b", "num_ctx": 4096, "num_predict": 350}, Journal(t))
            captured = []
            def api(path, payload=None):
                if path == "/api/chat":
                    captured.append(payload)
                    return {"done": True, "done_reason": "stop", "message": {"content": json.dumps({"requests_missing_information": True, "unsupported_claims": [], "inappropriate_requests": []})}}
                return {"models": []}
            client.api = api
            client.ask("validator", ClarificationReview, {"candidate_answer": "Размер?"})
            self.assertEqual(captured[0]["model"], "gemma3:4b")
            self.assertIn("УТОЧНЯЮЩИЙ ОТВЕТ", captured[0]["messages"][0]["content"])
            self.assertIn("requests_missing_information", captured[0]["format"]["properties"])
            events = (Path(t) / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn('"kind": "llm"', events)
            self.assertIn('"event": "span_end"', events)

    def test_ollama_http_error_keeps_server_detail(self):
        import io
        from urllib.error import HTTPError

        with tempfile.TemporaryDirectory() as t:
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b",
                             "num_ctx": 4096, "num_predict": 350}, Journal(t))

            class FailingOpener:
                def open(self, request, timeout):
                    raise HTTPError(request.full_url, 500, "Internal Server Error", {},
                                    io.BytesIO(b'{"error":"runner failed to load"}'))

            client.opener = FailingOpener()
            with self.assertRaisesRegex(RuntimeError, "runner failed to load"):
                client.api("/api/chat", {})

    def test_access_agent_loads_its_own_procedure(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b",
                             "num_ctx": 4096, "num_predict": 350}, Journal(t))
            captured = []

            def api(path, payload=None):
                if path == "/api/chat":
                    captured.append(payload)
                    return {"done": True, "done_reason": "stop", "message": {"content": json.dumps(
                        {"action": "check_course_access", "reason": "тест"})}}
                return {"models": []}

            client.api = api
            client.ask("diagnostician", decision_schema(["check_course_access"]), {"category": "access"})
            system_prompt = captured[0]["messages"][0]["content"]
            self.assertIn("# Проверка доступа к курсу", system_prompt)
            self.assertNotIn("# Диагностика загрузки", system_prompt)

    def test_primary_model_failure_survives_unload_failure(self):
        with tempfile.TemporaryDirectory() as t:
            journal = Journal(t)
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b",
                             "num_ctx": 4096, "num_predict": 350}, journal)

            def failing_api(path, data=None):
                if path == "/api/chat":
                    raise RuntimeError("первичная ошибка загрузки валидатора")
                raise RuntimeError("ошибка выгрузки")

            client.api = failing_api
            with self.assertRaisesRegex(RuntimeError, "первичная ошибка загрузки валидатора"):
                client.ask("validator", Verdict, {})
            log = (Path(t) / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("llm_error", log)
            self.assertIn("model_unload_error", log)

    def test_dual_residency_keeps_both_models_until_close(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b",
                             "num_ctx": 4096, "num_predict": 350,
                             "model_residency": "dual"}, Journal(tmp))
            resident = set()
            calls = []

            def api(path, data=None):
                calls.append((path, data))
                if path == "/api/ps":
                    return {"models": [{"name": name} for name in resident]}
                if path == "/api/generate":
                    resident.discard(data["model"])
                    return {"done": True}
                if path == "/api/chat":
                    resident.add(data["model"])
                    answer = ({"category": "upload", "reason": "test"} if data["model"].startswith("qwen")
                              else {"approved": True, "issues": []})
                    return {"done": True, "done_reason": "stop",
                            "message": {"content": json.dumps(answer)}}
                raise AssertionError(path)

            client.api = api
            client.ask("dispatcher", Route, {})
            client.ask("validator", Verdict, {})
            self.assertEqual(resident, {"qwen3.5:2b", "gemma3:4b"})
            self.assertEqual(len([path for path, _ in calls if path == "/api/generate"]), 0)
            self.assertEqual([data["keep_alive"] for path, data in calls if path == "/api/chat"], [-1, -1])
            client.close()
            self.assertFalse(resident)

    def test_dual_residency_detects_single_model_server_limit(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b",
                             "num_ctx": 4096, "num_predict": 350,
                             "model_residency": "dual"}, Journal(tmp))
            def api(path, data=None):
                if path == "/api/chat":
                    return {"done": True, "done_reason": "stop", "message": {"content": json.dumps(
                        {"approved": True, "issues": []})}}
                if path == "/api/ps":
                    return {"models": [{"name": "gemma3:4b"}]}
                return {"done": True}
            client.api = api
            with self.assertRaisesRegex(RuntimeError, "не удерживает обе модели"):
                client.ask("validator", Verdict, {})

    def test_dual_residency_stops_when_available_ram_is_low(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b",
                             "num_ctx": 4096, "num_predict": 350,
                             "model_residency": "dual", "dual_min_available_ram_gb": 1.5}, Journal(tmp))
            def api(path, data=None):
                if path == "/api/chat":
                    return {"done": True, "done_reason": "stop", "message": {"content": json.dumps(
                        {"category": "upload", "reason": "test"})}}
                return {"models": []}
            client.api = api
            with patch("support_agent.llm.available_system_ram_gb", return_value=0.5):
                with self.assertRaisesRegex(RuntimeError, "мало свободной ОЗУ"):
                    client.ask("dispatcher", Route, {})

    def test_grouped_residency_reuses_main_model_without_overlap(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            client = Ollama({"main_model": "qwen3.5:2b", "validator_model": "gemma3:4b",
                             "num_ctx": 4096, "num_predict": 350,
                             "model_residency": "grouped"}, Journal(tmp))
            resident, calls, peaks = set(), [], []

            def api(path, data=None):
                calls.append((path, data))
                if path == "/api/ps":
                    return {"models": [{"name": name} for name in resident]}
                if path == "/api/generate":
                    resident.discard(data["model"])
                    return {"done": True}
                if path == "/api/chat":
                    resident.add(data["model"])
                    peaks.append(len(resident))
                    answer = ({"category": "upload", "reason": "test"} if data["model"].startswith("qwen")
                              else {"approved": True, "issues": []})
                    return {"done": True, "done_reason": "stop",
                            "message": {"content": json.dumps(answer)}}
                raise AssertionError(path)

            client.api = api
            client.ask("dispatcher", Route, {})
            client.ask("dispatcher", Route, {})
            self.assertEqual(resident, {"qwen3.5:2b"})
            client.ask("validator", Verdict, {})
            self.assertEqual(resident, {"gemma3:4b"})
            client.ask("dispatcher", Route, {})
            self.assertEqual(resident, {"qwen3.5:2b"})
            self.assertEqual(peaks, [1, 1, 1, 1])
            self.assertEqual(len([path for path, _ in calls if path == "/api/generate"]), 2)
            client.close()
            self.assertFalse(resident)


if __name__ == "__main__":
    unittest.main()
