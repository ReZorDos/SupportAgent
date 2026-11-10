import hashlib
import sqlite3
import time
from pathlib import Path


class MemoryStore:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA busy_timeout = 3000")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge (
                category TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                version TEXT NOT NULL,
                content TEXT NOT NULL,
                loaded_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                file_extension TEXT NOT NULL,
                issue_code TEXT NOT NULL,
                resolution_kind TEXT NOT NULL,
                confirmed_at REAL NOT NULL,
                confirmation TEXT NOT NULL CHECK(confirmation = 'user_confirmed')
            );
            CREATE INDEX IF NOT EXISTS episodes_lookup
                ON episodes(category, file_extension, confirmed_at DESC);
        """)

    def close(self):
        self.connection.close()

    def sync_local_procedure(self, category, path, ttl_days=30):
        if category not in {"upload", "service", "access", "login", "materials", "grades"} or ttl_days <= 0:
            raise ValueError("Недопустимая процедура или срок")
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        version = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        existing = self.connection.execute(
            "SELECT version FROM knowledge WHERE category = ?", (category,)).fetchone()
        if not existing or existing[0] != version:
            now = time.time()
            self.connection.execute("""
                INSERT INTO knowledge(category, source, version, content, loaded_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(category) DO UPDATE SET
                    source=excluded.source, version=excluded.version, content=excluded.content,
                    loaded_at=excluded.loaded_at, expires_at=excluded.expires_at
            """, (category, path.name, version, content, now, now + ttl_days * 86400))
            self.connection.commit()
        return version

    def lookup(self, category, filename=""):
        if category not in {"upload", "service", "access", "login", "materials", "grades"}:
            return {"knowledge": [], "confirmed_episodes": []}
        now = time.time()
        row = self.connection.execute("""
            SELECT source, version, content, expires_at FROM knowledge
            WHERE category = ? AND expires_at > ?
        """, (category, now)).fetchone()
        knowledge = ([{"source": row[0], "version": row[1], "content": row[2],
                       "expires_at": row[3]}] if row else [])
        extension = Path(filename).suffix.lower() if category == "upload" else ""
        episodes = self.connection.execute("""
            SELECT issue_code, resolution_kind, confirmed_at FROM episodes
            WHERE category = ? AND file_extension = ?
            ORDER BY confirmed_at DESC, id DESC LIMIT 3
        """, (category, extension)).fetchall()
        return {"knowledge": knowledge,
                "confirmed_episodes": [{"issue_code": issue, "resolution_kind": remedy,
                                        "confirmed_at": when} for issue, remedy, when in episodes]}

    def is_stale(self, category):
        row = self.connection.execute("SELECT expires_at FROM knowledge WHERE category = ?",
                                      (category,)).fetchone()
        return bool(row and row[0] <= time.time())

    def confirm(self, result):
        if result.get("status") != "solution" or result.get("verdict", {}).get("approved") is not True:
            raise ValueError("Сохраняется только подтверждённое решение, одобренное валидатором")
        category = result.get("category")
        evidence = {item["id"]: item["data"] for item in result.get("evidence", [])}
        errors = set(evidence.get("check_file", {}).get("errors", []))
        if category == "upload" and errors == {"file_too_large"}:
            issue, remedy = "file_too_large", "compress_pdf"
        elif category == "upload" and errors == {"unsupported_format"}:
            issue, remedy = "unsupported_format", "save_as_pdf"
        elif category == "service" and evidence.get("get_service_status", {}).get("platform_service") == "unavailable":
            issue, remedy = "service_unavailable", "retry_later"
        elif category == "access" and evidence.get("check_course_access", {}).get("enrolled") is False:
            issue, remedy = "not_enrolled", "request_enrollment"
        elif category == "access" and evidence.get("check_course_access", {}).get("course_open") is False:
            issue, remedy = "course_not_open", "ask_instructor_opening"
        elif category == "login" and evidence.get("check_demo_account", {}).get("status") == "locked":
            issue, remedy = "demo_account_locked", "contact_demo_admin"
        elif category == "login" and evidence.get("check_demo_account", {}).get("login_service") == "unavailable":
            issue, remedy = "demo_login_unavailable", "retry_later"
        elif category == "materials" and evidence.get("check_material_status", {}).get("status") == "draft":
            issue, remedy = "material_unpublished", "contact_instructor"
        elif category == "materials" and evidence.get("check_material_status", {}).get("storage_status") == "unavailable":
            issue, remedy = "material_storage_unavailable", "retry_later"
        elif category == "grades" and evidence.get("check_grade_status", {}).get("status") in {"pending", "unpublished"}:
            issue = "grade_" + evidence["check_grade_status"]["status"]
            remedy = "wait_for_review" if issue == "grade_pending" else "wait_for_publication"
        else:
            raise ValueError("Нет однозначного подтверждённого шаблона решения")
        extension = Path(result.get("filename") or "").suffix.lower() if category == "upload" else ""
        with self.connection:
            self.connection.execute("""
                INSERT INTO episodes(category, file_extension, issue_code, resolution_kind,
                                     confirmed_at, confirmation)
                VALUES (?, ?, ?, ?, ?, 'user_confirmed')
            """, (category, extension, issue, remedy, time.time()))
        return {"category": category, "issue_code": issue, "resolution_kind": remedy}
