import re
from .models import Draft


def clarification_verdict(review):
    issues = list(review["unsupported_claims"]) + list(review["inappropriate_requests"])
    if not review["requests_missing_information"]:
        issues.append("Ответ не запрашивает недостающие сведения для диагностики.")
    return {"approved": not issues, "issues": issues}

def clarification_task(state):
    observations = {x["id"]: x["data"] for x in state.get("evidence", [])}
    return {
        "review_target": "Оцени ответ сотрудника, который просит пользователя дополнить сведения. Не проверяй, выполнил ли пользователь эту просьбу.",
        "customer_message": state["message"],
        "category": state.get("category", "upload"),
        "conversation": state.get("history", []),
        "known_metadata": {"filename": state.get("filename"), "size_mb": state.get("size_mb")},
        "course_code": state.get("course_code"),
        "demo_account_code": state.get("demo_account_code"),
        "material_code": state.get("material_code"),
        "submission_code": state.get("submission_code"),
        "missing_information": (observations.get("check_file", {}).get("missing", [])
                                + observations.get("check_course_access", {}).get("missing", [])
                                + observations.get("check_demo_account", {}).get("missing", [])
                                + observations.get("check_material_status", {}).get("missing", [])
                                + observations.get("check_grade_status", {}).get("missing", [])),
        "candidate_answer": state["draft"]["answer"],
        "evidence": state.get("evidence", []),
    }

def required_resolution_mode(state):
    if state.get("category") in {"materials", "grades"}:
        category = state["category"]
        tool = "check_material_status" if category == "materials" else "check_grade_status"
        observation = next((item["data"] for item in state.get("evidence", [])
                            if item["id"] == tool), None)
        if observation is None:
            return None
        status = observation.get("status")
        if category == "materials":
            return ("solution" if status == "draft" or observation.get("storage_status") == "unavailable"
                    else "clarification")
        return "solution" if status in {"pending", "unpublished"} else "clarification"
    if state.get("category") == "login":
        account = next((item["data"] for item in state.get("evidence", [])
                        if item["id"] == "check_demo_account"), None)
        if account is None:
            return None
        return ("solution" if account.get("status") == "locked" or
                account.get("login_service") == "unavailable" else "clarification")
    if state.get("category") == "access":
        access = next((item["data"] for item in state.get("evidence", [])
                       if item["id"] == "check_course_access"), None)
        if access is None:
            return None
        return ("solution" if access.get("enrolled") is False or
                access.get("course_open") is False else "clarification")
    if state.get("category") == "service":
        service = next((item["data"].get("platform_service")
                        for item in state.get("evidence", []) if item["id"] == "get_service_status"), None)
        return "solution" if service == "unavailable" else "clarification" if service == "available" else None
    if known_file_problem(state):
        return "solution"
    observations = {item["id"]: item["data"] for item in state.get("evidence", [])}
    check = observations.get("check_file")
    rules = observations.get("get_upload_rules")
    if check is None or rules is None:
        return None
    consistent = (rules.get("max_size_mb") == check.get("max_size_mb")
                  and rules.get("formats") == check.get("allowed_formats"))
    service = observations.get("get_service_status", {}).get("upload_service")


    if consistent and check.get("errors") == [] and service in (None, "available"):
        return "clarification"
    return None

def known_file_problem(state):
    evidence = {item["id"]: item["data"] for item in state.get("evidence", [])}
    check = evidence.get("check_file", {})
    rules = evidence.get("get_upload_rules", {})
    errors = set(check.get("errors", []))
    return bool(
        errors and errors.issubset({"file_too_large", "unsupported_format"})
        and not check.get("missing", ["unknown"])
        and rules.get("max_size_mb") == check.get("max_size_mb")
        and rules.get("formats") == check.get("allowed_formats")
        and evidence.get("get_service_status", {}).get("upload_service", "available") == "available"
    )


def evidence_conflict(state):
    evidence = {item["id"]: item["data"] for item in state.get("evidence", [])}
    rules = evidence.get("get_upload_rules")
    check = evidence.get("check_file")
    if rules and check and (rules.get("max_size_mb") != check.get("max_size_mb")
                            or rules.get("formats") != check.get("allowed_formats")):
        return True
    service = evidence.get("get_service_status", {}).get("upload_service")
    return bool(check and check.get("errors") and service == "unavailable")

def enforce_verdict(state, result):
    issues = list(result["issues"])
    known = {x["id"] for x in state.get("evidence", [])}
    ids = set(state["draft"]["evidence_ids"])
    if not ids.issubset(known) or (state["draft"]["mode"] == "solution" and not ids):
        issues.append("Отсутствуют или выдуманы ссылки на доказательства.")
    required = required_resolution_mode(state)
    if required and state["draft"]["mode"] != required:
        issues.append(f"Результаты инструментов требуют mode={required}. Если причина не установлена, нужно уточнить сведения, а не придумывать решение.")
    if known_file_problem(state) and not {"check_file", "get_upload_rules"}.issubset(ids):
        issues.append("Для установленной ошибки нужны ссылки check_file и get_upload_rules.")
    if (state.get("category") == "service" and required_resolution_mode(state) == "solution"
            and "get_service_status" not in ids):
        issues.append("Для подтверждённой недоступности сервиса нужна ссылка get_service_status.")
    if (state.get("category") == "access" and required_resolution_mode(state) == "solution"
            and "check_course_access" not in ids):
        issues.append("Для подтверждённого ограничения курса нужна ссылка check_course_access.")
    if (state.get("category") == "access" and required_resolution_mode(state) == "solution"
            and not re.search(r"симулятор|тестов", state["draft"]["answer"], re.IGNORECASE)):
        issues.append("Уточните, что зачисление проверено лишь для тестового профиля симулятора.")
    if state.get("category") == "login":
        answer = state["draft"]["answer"]
        account = next((item["data"] for item in state.get("evidence", [])
                        if item["id"] == "check_demo_account"), {})
        if (required == "solution" and ("check_demo_account" not in ids or
                not re.search(r"симулятор|тестов", answer, re.IGNORECASE))):
            issues.append("Подтвердите ссылкой проверку тестового профиля и назовите симулятор.")
        if (account.get("status") == "active" and account.get("login_service") == "available"
                and state["draft"]["mode"] == "clarification"
                and re.search(r"\b(?:уточните|укажите|сообщите|напишите)\b[^.!?]{0,65}"
                              r"\b(?:код\s+(?:тестового\s+)?(?:профиля|аккаунта)|"
                              r"какой\s+(?:именно\s+)?(?:профиль|аккаунт)|"
                              r"имя\s+(?:тестового\s+)?профиля)\b", answer, re.IGNORECASE)):
            issues.append("Код тестового профиля уже известен; спросите текст ошибки или этап входа.")
        if state["draft"]["mode"] == "clarification":
            if (not account.get("demo_account_code") and not re.search(
                    r"\b(?:уточните|укажите|сообщите|напишите|предоставьте)\b[^.!?]{0,65}"
                    r"\bкод\s+(?:тестового\s+)?(?:профиля|аккаунта)\b", answer,
                    re.IGNORECASE)):
                issues.append("Для проверки входа запросите неизвестный код тестового профиля.")


            if (account.get("status") is None and re.search(r"\bзаблокирован\w*\b", answer, re.IGNORECASE)
                    or account.get("status") == "active" and re.search(
                        r"(?<!не )\bзаблокирован\w*\b", answer, re.IGNORECASE)):
                issues.append("Блокировка не подтверждена проверкой тестового профиля.")
            if account.get("status") is None and re.search(
                    r"\b(?:активен|активна|активный|активная|разблокирован\w*)\b", answer,
                    re.IGNORECASE):
                issues.append("Состояние тестового профиля неизвестно; запросите его код.")
            if re.search(r"\b(?:ошибка|проблема)\b[^.!?]{0,45}\bне\s+связан\w*\s+с\s+доступност",
                         answer, re.IGNORECASE):
                issues.append("Проверка тестового профиля не исключает причину ошибки входа.")
        if re.search(r"\b(?:укажите|сообщите|пришлите|введите|отправьте|напишите)\b[^.!?]{0,70}"
                     r"\b(?:парол[ья]|код\s+подтверждения)\b", answer, re.IGNORECASE):
            issues.append("Нельзя запрашивать пароль или код подтверждения.")
        if re.search(r"\b(?:зачислен\w*|курс\w*)\b", answer, re.IGNORECASE):
            issues.append("Зачисление на курс не относится к проверке входа и здесь не установлено.")
    if state.get("category") in {"materials", "grades"}:
        category = state["category"]
        tool = "check_material_status" if category == "materials" else "check_grade_status"
        observation = next((item["data"] for item in state.get("evidence", [])
                            if item["id"] == tool), {})
        status = observation.get("status")
        answer = state["draft"]["answer"]
        if required == "solution" and (tool not in ids or not re.search(
                r"симулятор|тестов", answer, re.IGNORECASE)):
            issues.append("Решение должно ссылаться на проверку и обозначать учебный симулятор.")
        if state["draft"]["mode"] == "clarification":
            if status is None and re.search(
                    r"\b(?:опубликован\w*|не\s+опубликован\w*|скрыт\w*|проверя\w*|готов\w*)\b",
                    answer, re.IGNORECASE):
                issues.append("Без известного тестового кода статус материала или оценки неизвестен.")
            if category == "materials" and status == "published" and re.search(
                    r"\bне\s+опубликован\w*\b", answer, re.IGNORECASE):
                issues.append("Материал опубликован в симуляторе; не утверждайте обратное.")
            if category == "grades" and status == "published" and re.search(
                    r"\b(?:не\s+опубликован\w*|проверя\w*)\b", answer, re.IGNORECASE):
                issues.append("Оценка опубликована в симуляторе; нельзя заявлять обратное.")
            known_code = observation.get("material_code") if category == "materials" else observation.get("submission_code")
            if not known_code and re.search(
                    r"\b(?:код|он)\b[^.!?]{0,40}\bотсутствует\s+в\s+системе\b", answer,
                    re.IGNORECASE):
                issues.append("Код не сообщён; нельзя утверждать, что он отсутствует в системе.")
            if category == "materials" and not known_code and re.search(
                    r"\bне\s+найден\w*\s+материал\w*\b[^.!?]{0,50}\b(?:кодом|каталоге|системе)\b",
                    answer, re.IGNORECASE):
                issues.append("Код не указан; нельзя утверждать, что материалы с ним не найдены.")
            if known_code and status is not None and re.search(
                    r"\b(?:уточните|укажите|сообщите|напишите)\b[^.!?]{0,65}"
                    r"\bкод\s+(?:тестового\s+)?(?:материала|сдачи|работы)\b", answer, re.IGNORECASE):
                issues.append("Код уже известен; спросите текст ошибки или страницу.")
    access = next((item["data"] for item in state.get("evidence", [])
                   if item["id"] == "check_course_access"), {})
    if (state.get("category") == "access" and access.get("enrolled") is True
            and state["draft"]["mode"] == "clarification"
            and re.search(r"\b(?:уточните|укажите|сообщите|напишите)\b[^.!?]{0,65}"
                          r"\b(?:какой\s+(?:именно\s+)?курс|код\s+курса|название\s+курса)\b",
                          state["draft"]["answer"], re.IGNORECASE)):
        issues.append("Код курса уже известен; спросите текст ошибки или название задания.")
    if state.get("category") == "service" and required_resolution_mode(state) == "clarification":


        answer = state["draft"]["answer"].strip()
        if re.search(r"^(?:учебный\s+)?(?:сайт|сервис)\b[^.!?]{0,45}\b(?:недоступен|не\s+работает)\b",
                     answer, re.IGNORECASE):
            issues.append("Доступный симулятор не подтверждает утверждение о недоступности сайта.")
    rules = next((item["data"] for item in state.get("evidence", [])
                  if item["id"] == "get_upload_rules"), {})
    if (state["draft"]["mode"] == "solution" and rules.get("formats") == ["pdf"]
            and re.search(r"\b(?:друг(?:ом|ой|ие|их)|ином|иной)\s+формат(?:е|а|ы)?\b",
                          state["draft"]["answer"], re.IGNORECASE)):
        issues.append("Разрешён только PDF: нельзя предлагать другой формат.")
    answer = state["draft"]["answer"]
    if re.search(r"\b(?:предложите|рекомендуйте|сообщите\s+пользователю)\b", answer, re.IGNORECASE):
        issues.append("Ответьте непосредственно пользователю, а не инструктируйте сотрудника.")
    if state["draft"]["mode"] == "solution":
        if (state.get("category") == "upload" and re.search(
                r"\bсжа\w*\b[^.!?]{0,45}\bформат\w*\s+pdf\b", answer, re.IGNORECASE)):
            issues.append("Для смены формата сохраните документ как PDF; сжатие не меняет формат.")
        if state.get("category") == "upload" and re.search(r"\bпереизда\w*\b", answer, re.IGNORECASE):
            issues.append("Не предлагайте неподтверждённый способ переиздать работу.")
        if state.get("category") == "login":
            account = next((item["data"] for item in state.get("evidence", [])
                            if item["id"] == "check_demo_account"), {})
            if account.get("status") == "locked" and not re.search(
                    r"администратор|поддержк", answer, re.IGNORECASE):
                issues.append("При блокировке предложите обратиться к администратору, а не только уточнить ошибку.")
        if state.get("category") == "grades":
            grade = next((item["data"] for item in state.get("evidence", [])
                          if item["id"] == "check_grade_status"), {})
            if grade.get("status") == "unpublished" and re.search(
                    r"(?:ждите\s+(?:результата\s+)?проверки|"
                    r"после\s+(?:завершения|окончания)\s+проверки|"
                    r"дождитесь\s+(?:завершения|окончания)\s+проверки|"
                    r"(?:ещё|пока)\s+проверя\w*)", answer, re.IGNORECASE):
                issues.append("Работа уже проверена; ждите публикации оценки, а не проверки.")
    if state["draft"]["mode"] == "clarification":
        observations = {item["id"]: item["data"] for item in state.get("evidence", [])}
        if state.get("category") == "upload" and re.search(
                r"\b(?:ошибка|проблема)\s+не\s+связан\w*\s+с\s+(?:размер|формат)",
                answer, re.IGNORECASE):
            issues.append("Соответствие метаданных правилам не исключает причину ошибки.")
        code_by_category = {"login": ("check_demo_account", "demo_account_code"),
                            "materials": ("check_material_status", "material_code"),
                            "access": ("check_course_access", "course_code"),
                            "grades": ("check_grade_status", "submission_code")}
        if state.get("category") in code_by_category:
            tool, field = code_by_category[state["category"]]
            if not observations.get(tool, {}).get(field) and re.search(
                    r"\b(?:не\s+распознан\w*|не\s+найден\w*\s+в\s+(?:системе|каталоге))\b",
                    answer, re.IGNORECASE):
                issues.append("Код не указан; нельзя объявлять его нераспознанным.")
        if state.get("category") == "login" and re.search(
                r"\b(?:ошибка|проблема)\s+не\s+связан\w*\s+с\s+доступност", answer,
                re.IGNORECASE):
            issues.append("Активный тестовый профиль не исключает причину ошибки входа.")
        if state.get("category") == "login" and re.search(
                r"\bвход\s+(?:доступен|работает|успешен)\b", answer, re.IGNORECASE):
            issues.append("Статус тестового профиля не доказывает успешный вход пользователя.")
        if state.get("category") == "materials" and re.search(
                r"\bопубликован\w*\s+и\s+доступн\w*\b", answer, re.IGNORECASE):
            issues.append("Публикация и статус хранилища не доказывают, что материал открывается у пользователя.")
    if not result["approved"] and not issues:
        issues.append("Валидатор отклонил ответ без объяснения; повторно проверьте обоснованность и полноту рекомендации.")
    return {"approved": bool(result["approved"] and not issues), "issues": issues}

def bind_evidence(state, draft):
    model_ids = list(draft["evidence_ids"])
    policy_ids = ["check_file", "get_upload_rules"] if known_file_problem(state) else []
    bound = {**draft, "evidence_ids": list(dict.fromkeys(model_ids + policy_ids))}
    return bound, {"model_evidence_ids": model_ids, "policy_evidence_ids": policy_ids}

def login_clarification_fallback(state):
    account = next((item["data"] for item in state.get("evidence", [])
                    if item["id"] == "check_demo_account"), None)
    if account is None:
        return None
    if account.get("status") == "active":
        answer = ("Тестовый профиль симулятора активен. "
                  "Какой точный текст или код ошибки появляется при попытке входа?")
    elif account.get("status") is None:
        answer = "Уточните код тестового профиля DEMO_OK, DEMO_LOCKED или DEMO_OFFLINE и текст ошибки при входе."
    else:
        return None
    return {"mode": "clarification", "answer": answer, "evidence_ids": ["check_demo_account"]}

def catalog_clarification_fallback(state):
    category = state.get("category")
    if category not in {"materials", "grades"}:
        return None
    tool = "check_material_status" if category == "materials" else "check_grade_status"
    observation = next((item["data"] for item in state.get("evidence", [])
                        if item["id"] == tool), None)
    if observation is None:
        return None
    if category == "materials":
        answer = ("Какой точный текст ошибки появляется при открытии материала?"
                  if observation.get("status") == "published" else
                  "Уточните код тестового материала MAT101, MAT102 или MAT103 и текст ошибки при просмотре.")
    else:
        answer = ("Какой текст ошибки появляется на странице оценок?"
                  if observation.get("status") == "published" else
                  "Уточните код тестовой сдачи SUB101, SUB102 или SUB103 и текст ошибки.")
    return {"mode": "clarification", "answer": answer, "evidence_ids": [tool]}

def grounded_clarification_fallback(state):
    category = state.get("category")
    if category == "login":
        return login_clarification_fallback(state)
    if category in {"materials", "grades"}:
        return catalog_clarification_fallback(state)
    evidence = {item["id"]: item["data"] for item in state.get("evidence", [])}
    if category == "access" and "check_course_access" in evidence:
        access = evidence["check_course_access"]
        answer = ("Тестовый профиль зачислен на курс в симуляторе. "
                  "Уточните название задания и точный текст ошибки."
                  if access.get("enrolled") is True else
                  "Уточните код тестового курса и точный текст ошибки.")
        ids = ["check_course_access"]
    elif category == "service" and evidence.get("get_service_status", {}).get("platform_service") == "available":
        answer, ids = "Какой точный текст ошибки вы видите при открытии страницы?", ["get_service_status"]
    elif category == "upload" and "check_file" in evidence and "get_upload_rules" in evidence:
        missing = evidence["check_file"].get("missing", [])
        if missing:
            answer = "Уточните расширение файла и его размер в мегабайтах."
        else:
            answer = "Какой точный текст ошибки появляется после выбора файла?"
        ids = ["check_file", "get_upload_rules"]
    else:
        return None
    return {"mode": "clarification", "answer": answer, "evidence_ids": ids}

def grounded_solution_fallback(state):
    evidence = {item["id"]: item["data"] for item in state.get("evidence", [])}
    category = state.get("category")
    ids = []
    answer = None
    if category == "upload" and known_file_problem(state):
        check = evidence["check_file"]
        ids = ["check_file", "get_upload_rules"]
        if "file_too_large" in check["errors"]:
            answer = ("По указанным данным PDF превышает лимит 20 МБ. "
                      "Сожмите файл до 20 МБ, сохранив необходимое содержимое, и загрузите снова.")
        elif "unsupported_format" in check["errors"]:
            answer = "Формат файла не поддерживается симулятором: разрешён PDF. Сохраните работу как PDF и загрузите снова."
    elif category == "service" and evidence.get("get_service_status", {}).get("platform_service") == "unavailable":
        ids = ["get_service_status"]
        answer = "Учебный сервис симулятора сейчас недоступен. Попробуйте открыть страницу позже."
    elif category == "access":
        access = evidence.get("check_course_access", {})
        ids = ["check_course_access"]
        if access.get("enrolled") is False:
            answer = "Тестовый профиль симулятора не зачислен на этот курс. Обратитесь к преподавателю или администратору для проверки доступа."
        elif access.get("course_open") is False:
            answer = "Курс в учебном симуляторе ещё не открыт. Уточните срок открытия у преподавателя."
    elif category == "login":
        account = evidence.get("check_demo_account", {})
        ids = ["check_demo_account"]
        if account.get("status") == "locked":
            answer = "Тестовый профиль симулятора заблокирован. Обратитесь к администратору для проверки доступа."
        elif account.get("login_service") == "unavailable":
            answer = "Тестовый сервис входа симулятора недоступен. Повторите попытку позже."
    elif category == "materials":
        material = evidence.get("check_material_status", {})
        ids = ["check_material_status"]
        if material.get("status") == "draft":
            answer = "Материал в учебном симуляторе ещё не опубликован. Уточните срок публикации у преподавателя."
        elif material.get("storage_status") == "unavailable":
            answer = "Учебное хранилище материалов симулятора недоступно. Попробуйте открыть материал позже."
    elif category == "grades":
        grade = evidence.get("check_grade_status", {})
        ids = ["check_grade_status"]
        if grade.get("status") == "pending":
            answer = "Тестовая работа в симуляторе ещё проверяется. Дождитесь окончания проверки."
        elif grade.get("status") == "unpublished":
            answer = "Оценка тестовой работы в симуляторе ещё не опубликована. Дождитесь публикации или уточните срок у преподавателя."
    if answer is None or not set(ids).issubset(evidence):
        return None
    return {"mode": "solution", "answer": answer, "evidence_ids": ids}

def adjudicate_grounded_fallback(state, llm_verdict):
    if llm_verdict["approved"] or not state.get("fallback_kind"):
        return None
    required = required_resolution_mode(state)
    canonical = (grounded_solution_fallback(state) if required == "solution" else
                 grounded_clarification_fallback(state) if required == "clarification" else None)
    if canonical is None or state["draft"] != canonical:
        return None
    evidence = {item["id"]: item["data"] for item in state.get("evidence", [])}
    if any(evidence.get(key, {}).get("result_available") is False
           for key in canonical["evidence_ids"]):
        return None
    policy = enforce_verdict(state, {"approved": True, "issues": []})
    if not policy["approved"]:
        return None
    return {"approved": True, "issues": []}
