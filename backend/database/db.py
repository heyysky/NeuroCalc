"""
NeuroCalc Database Module
SQLite-only, lightweight, no ORM overhead
"""

import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "neurocalc.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                equation  TEXT NOT NULL,
                result    TEXT,
                steps     TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    logger.info(f"✓ SQLite DB ready: {DB_PATH}")


def save_history(equation: str, result: str, steps: List[Dict]) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO history (equation, result, steps) VALUES (?, ?, ?)",
            (equation, result, json.dumps(steps)),
        )
        conn.commit()
        return cur.lastrowid


def get_history(limit: int = 20) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, equation, result, steps, created_at FROM history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "equation": row["equation"],
            "result": row["result"],
            "steps": json.loads(row["steps"] or "[]"),
            "created_at": row["created_at"],
        })
    return result


def delete_history(entry_id: int) -> bool:
    with get_connection() as conn:
        conn.execute("DELETE FROM history WHERE id = ?", (entry_id,))
        conn.commit()
    return True


def clear_history():
    with get_connection() as conn:
        conn.execute("DELETE FROM history")
        conn.commit()
