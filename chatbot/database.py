"""
database.py
-----------
SQLite database module for persisting conversation history.

Stores every message exchanged between the user and the chatbot,
tagged by session ID, so conversations can be reviewed or analysed.
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Database path — sits inside the data/ directory
# ---------------------------------------------------------------------------
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
_DB_PATH = os.path.join(_DATA_DIR, "conversations.db")


def _get_connection() -> sqlite3.Connection:
    """Open (and create if needed) the SQLite database."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
    return conn


def init_db() -> None:
    """Create the conversations table if it does not exist."""
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            role        TEXT    NOT NULL,   -- 'user' or 'assistant'
            message     TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, message: str) -> None:
    """
    Save a single chat message to the database.

    Parameters
    ----------
    session_id : unique identifier for this conversation session
    role       : 'user' or 'assistant'
    message    : the message text
    """
    conn = _get_connection()
    conn.execute(
        "INSERT INTO conversations (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, message, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(session_id: str) -> list[dict]:
    """
    Retrieve the full conversation history for a session.

    Returns a list of dicts: [{'role': ..., 'message': ..., 'timestamp': ...}, ...]
    """
    conn = _get_connection()
    cursor = conn.execute(
        "SELECT role, message, timestamp FROM conversations WHERE session_id = ? ORDER BY id ASC",
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [{"role": r[0], "message": r[1], "timestamp": r[2]} for r in rows]


def clear_session(session_id: str) -> None:
    """Delete all messages for a given session."""
    conn = _get_connection()
    conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def get_all_sessions() -> list[str]:
    """Return a list of all unique session IDs in the database."""
    conn = _get_connection()
    cursor = conn.execute("SELECT DISTINCT session_id FROM conversations ORDER BY session_id")
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return sessions


# Initialise the database on first import
init_db()
