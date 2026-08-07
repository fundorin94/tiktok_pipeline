import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    topic TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    sensitivity_flag TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(id)
);

CREATE TABLE IF NOT EXISTS api_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases(id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path):
        self.path = str(path)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_case(self, case_id: str, topic: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO cases (id, topic, status, created_at) VALUES (?, ?, 'new', ?)",
                (case_id, topic, _now()),
            )
            # An explicitly given topic must win. INSERT OR IGNORE keeps the
            # stored one, so re-running an existing case with a new topic
            # silently researched the old subject -- once producing an
            # entirely different killer's case than the one asked for.
            if topic:
                conn.execute("UPDATE cases SET topic = ? WHERE id = ?", (topic, case_id))

    def get_case(self, case_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()

    def update_case_status(self, case_id: str, status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE cases SET status = ? WHERE id = ?", (status, case_id))

    def set_sensitivity_flag(self, case_id: str, flag: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE cases SET sensitivity_flag = ? WHERE id = ?", (flag, case_id))

    def log_stage(self, case_id: str, stage: str, status: str, message: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO stage_log (case_id, stage, status, message, timestamp) VALUES (?, ?, ?, ?, ?)",
                (case_id, stage, status, message, _now()),
            )

    def get_stage_history(self, case_id: str, stage: str | None = None) -> list[sqlite3.Row]:
        with self._connect() as conn:
            if stage:
                return conn.execute(
                    "SELECT * FROM stage_log WHERE case_id = ? AND stage = ? ORDER BY id",
                    (case_id, stage),
                ).fetchall()
            return conn.execute(
                "SELECT * FROM stage_log WHERE case_id = ? ORDER BY id", (case_id,)
            ).fetchall()

    def log_usage(self, case_id: str, stage: str, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_usage (case_id, stage, model, input_tokens, output_tokens, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, stage, model, input_tokens, output_tokens, _now()),
            )

    def get_usage(self, case_id: str | None = None) -> list[sqlite3.Row]:
        with self._connect() as conn:
            if case_id:
                return conn.execute(
                    "SELECT * FROM api_usage WHERE case_id = ? ORDER BY id", (case_id,)
                ).fetchall()
            return conn.execute("SELECT * FROM api_usage ORDER BY id").fetchall()
