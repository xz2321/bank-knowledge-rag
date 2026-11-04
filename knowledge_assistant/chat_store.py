"""SQLite 会话存储：会话列表 + 消息（含引用），刷新/重启都不丢。"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("CHAT_DB", str(ROOT / "knowledge_base_output" / "chat_sessions.db")))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init() -> None:
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '新对话',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                citations TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            """
        )


def create_session(title: str = "新对话") -> str:
    session_id = uuid.uuid4().hex[:12]
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions(id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
    return session_id


def rename_session(session_id: str, title: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title[:60], time.time(), session_id),
        )


def touch(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time(), session_id))


def add_message(session_id: str, role: str, content: str, citations: list | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, citations, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json.dumps(citations or [], ensure_ascii=False), time.time()),
        )
    touch(session_id)


def get_messages(session_id: str, limit: int | None = None) -> list[dict]:
    query = "SELECT id, role, content, citations, created_at FROM messages WHERE session_id = ? ORDER BY id"
    with _connect() as conn:
        rows = conn.execute(query, (session_id,)).fetchall()
    items = [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "citations": json.loads(row["citations"] or "[]"),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return items[-limit:] if limit else items


def list_sessions(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count,
                   (SELECT content FROM messages m WHERE m.session_id = s.id AND m.role = 'user'
                    ORDER BY m.id DESC LIMIT 1) AS last_question
            FROM sessions s
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
