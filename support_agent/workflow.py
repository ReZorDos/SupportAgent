import hashlib
import json
import math
import time
import uuid
from langgraph.graph import StateGraph, START, END
from .models import (State, Route, Draft, Verdict, ClarificationReview, SolutionDraft, ClarificationDraft, decision_schema)
from .policy import (clarification_task, clarification_verdict, required_resolution_mode, grounded_solution_fallback, grounded_clarification_fallback, bind_evidence, adjudicate_grounded_fallback, enforce_verdict, evidence_conflict)
from .tools import execute_tool
from .settings import ROOT


def build_graph(client, journal, config, memory_store=None):
    def dispatcher(s):
        result = client.ask("dispatcher", Route, {"message": s["message"],
                                               "conversation": s.get("history", [])})
        journal.emit("route", **result)
        return {"category": result["category"]}

    def recall(s):
        span_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = "ok"
        journal.emit("span_start", span_id=span_id, parent_span_id=journal.turn_id,
                     kind="memory", name="lookup")
        try:
            context = memory_store.lookup(s["category"], s.get("filename", "")) if memory_store else {
                "knowledge": [], "confirmed_episodes": []}
            if memory_store:
                journal.emit("memory_lookup", category=s["category"],
                             knowledge_versions=[item["version"] for item in context["knowledge"]],
                             confirmed_count=len(context["confirmed_episodes"]))
        except Exception:
            status = "error"
            raise
        finally:
            journal.emit("span_end", span_id=span_id, kind="memory", name="lookup",
                         status=status, duration_seconds=round(time.perf_counter() - started, 3))
        available = not memory_store or not memory_store.is_stale(s["category"])
        if not available:
            journal.emit("knowledge_expired", category=s["category"])
        return {"memory_context": context, "procedure_available": available}

    def diagnostician(s):
        used = {x["id"] for x in s.get("evidence", [])}
        tools = ({"service": ("get_service_status",), "access": ("check_course_access",),
                  "login": ("check_demo_account",), "materials": ("check_material_status",),
                  "grades": ("check_grade_status",)}
                 .get(s["category"], ("get_upload_rules", "check_file", "get_service_status")))
        allowed = [x for x in tools if x not in used]
        if s.get("tool_calls", 0) >= config["max_tool_calls"]:
            return {"action": "finish"}

        if used:
            allowed.append("finish")
        result = client.ask("diagnostician", decision_schema(allowed), {"message": s["message"],
                            "category": s["category"],
                            "memory_context": s.get("memory_context", {}),
                            "conversation": s.get("history", []),
                            "filename": s.get("filename"), "size_mb": s.get("size_mb"),
                            "course_code": s.get("course_code"),
                            "demo_account_code": s.get("demo_account_code"),
                            "material_code": s.get("material_code"),
                            "submission_code": s.get("submission_code"),
                            "evidence": s.get("evidence", []), "allowed_actions": allowed})
        if result["action"] not in allowed:
            raise RuntimeError(f"Диагност выбрал {result['action']!r}; допустимы: {allowed}")
        journal.emit("diagnostic_decision", **result)
        return {"action": result["action"]}

    def tool(s):
        span_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = "ok"
        journal.emit("span_start", span_id=span_id, parent_span_id=journal.turn_id,
                     kind="tool", name=s["action"])
        try:
            observation = execute_tool(s["action"], {**s,
                "simulator_service_status": config.get("simulator_service_status", "available"),
                "simulator_tool_failures": config.get("simulator_tool_failures", [])})
            journal.emit("tool_result", observation=observation)
        except Exception as exc:
            status = "error"
            journal.emit("tool_error", action=s["action"], message=str(exc))
            observation = {"id": s["action"], "data": {"source": "tool_error",
                           "error": type(exc).__name__, "result_available": False}}
        finally:
            journal.emit("span_end", span_id=span_id, kind="tool", name=s["action"],
                         status=status, duration_seconds=round(time.perf_counter() - started, 3))
        observations = s.get("evidence", []) + [observation]
        conflict = status == "ok" and evidence_conflict({**s, "evidence": observations})
        if conflict:
            journal.emit("evidence_conflict", category=s["category"])
        return {"evidence": observations, "tool_calls": s.get("tool_calls", 0) + 1,
                "action": "tool_failed" if status == "error" else "conflict" if conflict else s["action"]}

    def resolver(s):
        required = required_resolution_mode(s)
        journal.emit("resolution_policy", required_mode=required)
        schema = {"solution": SolutionDraft, "clarification": ClarificationDraft}.get(required, Draft)
        data = {"message": s["message"],
                           "category": s["category"],
                           "memory_context": s.get("memory_context", {}),
                           "conversation": s.get("history", []),
                           "required_mode": required,
                           "filename": s.get("filename"), "size_mb": s.get("size_mb"),
                           "course_code": s.get("course_code"),
                           "demo_account_code": s.get("demo_account_code"),
                           "material_code": s.get("material_code"),
                           "submission_code": s.get("submission_code"),
                           "evidence": s.get("evidence", []), "previous_draft": s.get("draft"),
                           "feedback": s.get("verdict", {}).get("issues", [])}
        rejected = s.get("verdict", {}).get("approved") is False
        fallback = (grounded_solution_fallback(s) if required == "solution" and rejected else
                    grounded_clarification_fallback(s) if required == "clarification" and rejected else None)
        if fallback is not None:
            journal.emit("solution_fallback" if fallback["mode"] == "solution" else "clarification_fallback",
                         reason="rejected_draft", evidence_ids=fallback["evidence_ids"])
            result = fallback
        else:
            result = client.ask("resolver", schema, data)
        result, binding = bind_evidence(s, result)
        journal.emit("evidence_binding", **binding)
        return {"draft": result, "evidence_binding": binding,
                "fallback_kind": result["mode"] if fallback is not None else None,
                "revisions": s.get("revisions", 0) + 1}

    def validator(s):
        if s["draft"]["mode"] == "clarification":
            review = client.ask("validator", ClarificationReview, clarification_task(s))
            journal.emit("clarification_review", **review)
            result = clarification_verdict(review)
        else:
            result = client.ask("validator", Verdict, {"message": s["message"],
                               "category": s["category"],
                               "memory_context": s.get("memory_context", {}),
                               "conversation": s.get("history", []),
                               "filename": s.get("filename"), "size_mb": s.get("size_mb"),
                               "course_code": s.get("course_code"),
                               "demo_account_code": s.get("demo_account_code"),
                               "material_code": s.get("material_code"),
                               "submission_code": s.get("submission_code"),
                               "evidence": s.get("evidence", []), "draft": s["draft"],
                               "required_mode": required_resolution_mode(s),
                               "evidence_binding": s.get("evidence_binding", {})})
        original = result
        adjudicated = adjudicate_grounded_fallback(s, original)
        if adjudicated is not None:
            journal.emit("judge_disagreement", judge_issues=original["issues"],
                         disposition="accepted_by_exact_template_and_policy")
            result = adjudicated
        else:
            result = enforce_verdict(s, original)
        journal.emit("validation", **result)
        return {"verdict": result}

    def after_validation(s):
        if s["verdict"]["approved"]:
            return "escalate" if s["draft"]["mode"] == "escalation" else "deliver"
        if s["revisions"] <= config["max_revisions"]:
            return "resolver"
        journal.emit("validator_retry_exhausted", revisions=s["revisions"])
        return "escalate"

    def deliver(s):
        return {"final": s["draft"]["answer"], "status": s["draft"]["mode"]}

    def escalate(s):
        journal.emit("procedure_loaded", category="escalation", role="graph",
                     version=hashlib.sha256((ROOT / "skills" / "escalation.md").read_bytes()).hexdigest()[:16])
        path = journal.directory / "ticket.json"
        path.write_text(json.dumps({"message": s["message"], "conversation": s.get("history", []),
                                   "evidence": s.get("evidence", []),
                                   "verdict": s.get("verdict"), "category": s.get("category"),
                                   "external_send": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        journal.emit("local_ticket", path=str(path))
        return {"final": "Нужен дополнительный разбор. Создана локальная учебная заявка; настоящему оператору она не отправлена.", "status": "escalation"}

    graph = StateGraph(State)
    for name, fn in {"dispatcher": dispatcher, "recall": recall, "diagnostician": diagnostician, "tool": tool,
                     "resolver": resolver, "validator": validator, "deliver": deliver, "escalate": escalate}.items():
        graph.add_node(name, fn)
    graph.add_edge(START, "dispatcher")
    graph.add_conditional_edges("dispatcher", lambda s: "recall" if s["category"] in {"upload", "service", "access", "login", "materials", "grades"} else "escalate", ["recall", "escalate"])
    graph.add_conditional_edges("recall", lambda s: "diagnostician" if s["procedure_available"] else "escalate",
                                ["diagnostician", "escalate"])
    graph.add_conditional_edges("diagnostician", lambda s: "resolver" if s["action"] == "finish" else "tool", ["resolver", "tool"])
    graph.add_conditional_edges("tool", lambda s: "escalate" if s["action"] in {"tool_failed", "conflict"} else "diagnostician",
                                ["escalate", "diagnostician"])
    graph.add_edge("resolver", "validator")
    graph.add_conditional_edges("validator", after_validation, ["resolver", "deliver", "escalate"])
    graph.add_edge("deliver", END)
    graph.add_edge("escalate", END)
    return graph.compile()

def run_turn(graph, journal, message, filename, size, history, course_code="", demo_account_code="",
             material_code="", submission_code=""):
    if not message or len(message) > 1500:
        raise ValueError("Введите сообщение от 1 до 1500 символов")
    if size is not None and (not math.isfinite(size) or size < 0):
        raise ValueError("Размер должен быть конечным неотрицательным числом")
    previous_turn = journal.turn_id
    journal.turn_id = uuid.uuid4().hex
    started = time.perf_counter()
    status = "ok"
    journal.emit("span_start", span_id=journal.turn_id, parent_span_id=None,
                 kind="turn", name="support_request")
    try:
        result = graph.invoke(
            {"message": message, "history": history[-8:], "filename": filename,
             "course_code": course_code, "demo_account_code": demo_account_code,
             "material_code": material_code, "submission_code": submission_code,
             "size_mb": size, "evidence": [], "tool_calls": 0, "revisions": 0},
            {"recursion_limit": 35})
        result["trace_turn_id"] = journal.turn_id
        journal.emit("turn_end", outcome=result["status"])
        return result
    except Exception as exc:
        status = "error"
        journal.emit("turn_error", error_type=type(exc).__name__)
        raise
    finally:
        journal.emit("span_end", span_id=journal.turn_id, kind="turn",
                     name="support_request", status=status,
                     duration_seconds=round(time.perf_counter() - started, 3))
        journal.turn_id = previous_turn
