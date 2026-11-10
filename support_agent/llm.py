import hashlib
import json
import math
import os
import sys
import time
import uuid
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import HTTPError
from urllib.parse import urlsplit
from .models import ClarificationReview
from .settings import ROOT, available_system_ram_gb


class Ollama:
    def __init__(self, config, journal):
        self.config, self.journal = config, journal
        endpoint = os.environ.get("SUPPORT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or
                not parsed.port or parsed.username or parsed.password or
                parsed.path or parsed.query or parsed.fragment):
            raise ValueError("SUPPORT_OLLAMA_URL: допустим только локальный адрес с портом")
        self.endpoint = endpoint
        self.opener = build_opener(ProxyHandler({}))
        self.residency = config.get("model_residency", "single")
        if self.residency not in {"single", "grouped", "dual"}:
            raise ValueError("model_residency должен быть single, grouped или dual")
        self._active_model = None

    def api(self, path, data=None):
        body = None if data is None else json.dumps(data, ensure_ascii=False).encode("utf-8")
        req = Request(self.endpoint + path, data=body,
                      headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=180) as response:
                value = json.load(response)
        except HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Ollama {path}: HTTP {exc.code}: {detail[:1000]}") from exc
        if "error" in value:
            raise RuntimeError(value["error"])
        return value

    def prepare(self):
        expected = {self.config["main_model"], self.config["validator_model"]}
        tags = {x["name"]: x for x in self.api("/api/tags")["models"]}
        if not expected.issubset(tags):
            raise RuntimeError("Не скачаны модели: " + ", ".join(expected - tags.keys()))
        families = [tags[m].get("details", {}).get("family") for m in (self.config["main_model"], self.config["validator_model"])]
        if not all(families) or families[0] == families[1]:
            raise RuntimeError("Основная модель и валидатор должны быть из разных подтверждённых семейств.")
        running = self.api("/api/ps").get("models", [])
        if running:
            raise RuntimeError("Перед запуском выгрузите модели: " + ", ".join(x["name"] for x in running) + ". Используйте ollama stop ИМЯ.")
        self.journal.emit("startup", config=self.config, models={m: tags[m] for m in expected})

    def close(self):
        if self.residency == "single":
            return
        errors = []
        for model in (self.config["main_model"], self.config["validator_model"]):
            try:
                self.api("/api/generate", {"model": model, "stream": False, "keep_alive": 0})
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                self.journal.emit("model_unload_error", role="shutdown", model=model, message=str(exc))
        if not errors:
            try:
                running = {m["name"] for m in self.api("/api/ps").get("models", [])}
                remaining = running.intersection({self.config["main_model"], self.config["validator_model"]})
                if remaining:
                    errors.append("Остались загружены: " + ", ".join(sorted(remaining)))
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("Не удалось выгрузить модели: " + "; ".join(errors))
        self.journal.emit("models_unloaded", mode=self.residency)

    def _switch_to(self, model):
        if self.residency != "grouped" or self._active_model in (None, model):
            return
        previous = self._active_model
        self.api("/api/generate", {"model": previous, "stream": False, "keep_alive": 0})
        if any(m["name"] == previous for m in self.api("/api/ps").get("models", [])):
            raise RuntimeError("Не удалось освободить память перед переключением моделей")
        self._active_model = None
        self.journal.emit("model_switch", unloaded=previous, next_model=model)

    def ask(self, role, schema, data):
        model = self.config["validator_model"] if role == "validator" else self.config["main_model"]
        policy = (ROOT / "PURPOSE.md").read_text(encoding="utf-8")
        policy += "\n" + (ROOT / "BEHAVIOR.md").read_text(encoding="utf-8")
        if role in {"diagnostician", "resolver", "validator"}:
            policy += "\n" + (ROOT / "MEMORY_POLICY.md").read_text(encoding="utf-8")
        if role == "diagnostician":
            policy += "\n" + (ROOT / "TOOL_POLICY.md").read_text(encoding="utf-8")
        prompt_name = "validator_clarification" if role == "validator" and schema is ClarificationReview else role
        prompt = (ROOT / "prompts" / f"{prompt_name}.md").read_text(encoding="utf-8")
        if role in {"diagnostician", "resolver"}:

            skill = data.get("category", "upload")
            if skill in {"upload", "service", "access", "login", "materials", "grades"}:
                procedure = (ROOT / "skills" / f"{skill}.md").read_text(encoding="utf-8")
                prompt += "\n" + procedure
                self.journal.emit("procedure_loaded", category=skill, role=role,
                                  version=hashlib.sha256(procedure.encode("utf-8")).hexdigest()[:16])
        prompt_version = hashlib.sha256((policy + "\n" + prompt).encode("utf-8")).hexdigest()[:16]
        payload = {"model": model, "stream": False,
                   "keep_alive": -1 if self.residency in {"grouped", "dual"} else 0,
                   "format": schema.model_json_schema(),
                   "options": {"num_ctx": self.config["num_ctx"], "num_predict": self.config["num_predict"], "temperature": 0, "seed": 42},
                   "messages": [{"role": "system", "content": policy + "\n" + prompt},
                                {"role": "user", "content": json.dumps(data, ensure_ascii=False)}]}
        if model.startswith("qwen"):
            payload["think"] = False
        started = time.perf_counter()
        span_id = uuid.uuid4().hex
        status = "ok"
        self.journal.emit("span_start", span_id=span_id, parent_span_id=self.journal.turn_id,
                          kind="llm", name=role, model=model, prompt_version=prompt_version)
        self.journal.emit("agent_start", role=role, model=model, prompt_version=prompt_version)
        try:
            self._switch_to(model)
            result = self.api("/api/chat", payload)
            if self.residency == "grouped":
                self._active_model = model
            self.journal.emit("llm_result", role=role, model=model, prompt_version=prompt_version, request=payload,
                              response=result, wall_seconds=time.perf_counter() - started)
            if not result.get("done") or result.get("done_reason") == "length":
                raise RuntimeError("Незавершённый ответ модели")
            if self.residency == "dual" and role == "validator":
                running = {m["name"] for m in self.api("/api/ps").get("models", [])}
                expected = {self.config["main_model"], self.config["validator_model"]}
                self.journal.emit("model_residency", mode="dual", models=sorted(running & expected))
                if not expected.issubset(running):
                    raise RuntimeError("Ollama не удерживает обе модели одновременно; "
                                       "вернитесь к model_residency=single или проверьте лимит моделей и ОЗУ")
            if self.residency == "dual":
                available = available_system_ram_gb()
                threshold = self.config.get("dual_min_available_ram_gb", 1.5)
                if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
                    raise ValueError("dual_min_available_ram_gb должен быть неотрицательным числом")
                if available is not None:
                    self.journal.emit("memory_guard", available_ram_gb=round(available, 2),
                                      min_available_ram_gb=threshold)
                    if available < threshold:
                        raise RuntimeError("Для удержания обеих моделей осталось слишком мало свободной ОЗУ; "
                                           "используйте model_residency=single")
            return schema.model_validate_json(result["message"]["content"]).model_dump()
        except Exception as exc:
            status = "error"
            self.journal.emit("llm_error", role=role, model=model,
                              wall_seconds=time.perf_counter() - started, message=str(exc))
            raise
        finally:


            primary_failed = sys.exc_info()[0] is not None
            try:
                if self.residency == "single":
                    self.api("/api/generate", {"model": model, "stream": False, "keep_alive": 0})
                    if any(m["name"] == model for m in self.api("/api/ps").get("models", [])):
                        raise RuntimeError("Не удалось подтвердить выгрузку модели")
            except Exception as unload_exc:
                status = "error"
                self.journal.emit("model_unload_error", role=role, model=model, message=str(unload_exc))
                if not primary_failed:
                    raise
            finally:
                self.journal.emit("span_end", span_id=span_id, kind="llm", name=role,
                                  status=status, duration_seconds=round(time.perf_counter() - started, 3))
