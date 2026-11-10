import math
from pathlib import Path


def execute_tool(name, state):
    if name in state.get("simulator_tool_failures", []):
        raise RuntimeError("Тестовый отказ инструмента " + name)
    if name == "get_upload_rules":
        data = {"source": "simulator_rules_v1", "formats": ["pdf"], "max_size_mb": 20}
    elif name == "get_service_status":
        status = state.get("simulator_service_status", "available")
        if status not in {"available", "unavailable"}:
            raise ValueError("Недопустимое состояние симулятора")
        data = {"source": "simulator", "upload_service": status, "platform_service": status}
    elif name == "check_course_access":
        code = (state.get("course_code") or "").strip().upper()
        if len(code) > 30:
            raise ValueError("Слишком длинный код курса")
        roster = {"CS101": (True, True), "CS202": (False, None), "CS303": (True, False)}
        data = {"source": "simulator_demo_roster_v1", "course_code": code,
                "enrolled": roster[code][0] if code in roster else None,
                "course_open": roster[code][1] if code in roster else None,
                "missing": ["код курса"] if not code else [],
                "unknown_course": bool(code and code not in roster), "actual_account_read": False}
    elif name == "check_demo_account":
        code = (state.get("demo_account_code") or "").strip().upper()
        if len(code) > 30:
            raise ValueError("Слишком длинный код тестового профиля")
        roster = {"DEMO_OK": ("active", "available"),
                  "DEMO_LOCKED": ("locked", "available"),
                  "DEMO_OFFLINE": ("active", "unavailable")}
        data = {"source": "simulator_demo_accounts_v1", "demo_account_code": code,
                "status": roster[code][0] if code in roster else None,
                "login_service": roster[code][1] if code in roster else None,
                "missing": ["код тестового профиля"] if not code else [],
                "unknown_account": bool(code and code not in roster), "actual_account_read": False}
    elif name == "check_material_status":
        code = (state.get("material_code") or "").strip().upper()
        if len(code) > 30:
            raise ValueError("Слишком длинный код тестового материала")
        catalog = {"MAT101": ("published", "available"),
                   "MAT102": ("draft", "available"),
                   "MAT103": ("published", "unavailable")}
        data = {"source": "simulator_material_catalog_v1", "material_code": code,
                "status": catalog[code][0] if code in catalog else None,
                "storage_status": catalog[code][1] if code in catalog else None,
                "missing": ["код тестового материала"] if not code else [],
                "unknown_material": bool(code and code not in catalog), "actual_platform_read": False}
    elif name == "check_grade_status":
        code = (state.get("submission_code") or "").strip().upper()
        if len(code) > 30:
            raise ValueError("Слишком длинный код тестовой сдачи")
        catalog = {"SUB101": "pending", "SUB102": "unpublished", "SUB103": "published"}
        data = {"source": "simulator_grade_catalog_v1", "submission_code": code,
                "status": catalog.get(code), "missing": ["код тестовой сдачи"] if not code else [],
                "unknown_submission": bool(code and code not in catalog), "actual_platform_read": False}
    elif name == "check_file":
        filename, size = state.get("filename", ""), state.get("size_mb")
        ext = Path(filename).suffix.lower().lstrip(".")
        errors = []
        missing = []
        if not ext:
            missing.append("расширение файла")
        elif ext != "pdf":
            errors.append("unsupported_format")
        if size is None:
            missing.append("размер файла")
        elif not math.isfinite(size) or size < 0:
            raise ValueError("Недопустимый размер файла")
        elif size > 20:
            errors.append("file_too_large")
        data = {"source": "user_reported_metadata", "filename": filename, "size_mb": size,
                "allowed_formats": ["pdf"], "max_size_mb": 20, "errors": errors,
                "missing": missing, "actual_file_read": False}
    else:
        raise ValueError("Инструмент не разрешён")
    return {"id": name, "data": data}
