"""
database.py
SQLite data layer for job management and player tracking records.
"""

import sqlite3
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "kabaddi.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id              TEXT PRIMARY KEY,
            input_filename  TEXT NOT NULL,
            input_path      TEXT,
            output_path     TEXT,
            status          TEXT NOT NULL DEFAULT 'queued',
            progress        INTEGER DEFAULT 0,
            total_frames    INTEGER DEFAULT 0,
            processed_frames INTEGER DEFAULT 0,
            elapsed_seconds REAL DEFAULT 0,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS players (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT NOT NULL,
            track_id    INTEGER NOT NULL,
            official_id TEXT NOT NULL,
            team        INTEGER NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(id)
        );
        """)


def create_job(input_filename: str) -> str:
    """Create a new job record and return its ID."""
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    with _conn() as con:
        con.execute(
            """INSERT INTO jobs (id, input_filename, status, created_at, updated_at)
               VALUES (?, ?, 'queued', ?, ?)""",
            (job_id, input_filename, now, now),
        )
    return job_id


def update_job(job_id: str, **kwargs) -> None:
    """Update one or more fields on a job row."""
    if not kwargs:
        return
    kwargs["updated_at"] = datetime.utcnow().isoformat()
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [job_id]
    with _conn() as con:
        con.execute(f"UPDATE jobs SET {fields} WHERE id = ?", values)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs() -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    return [dict(r) for r in rows]


def save_player_record(job_id: str, track_id: int, official_id: str, team: int) -> None:
    with _conn() as con:
        # Upsert: delete existing record for this job+track, then insert
        con.execute(
            "DELETE FROM players WHERE job_id = ? AND track_id = ?",
            (job_id, track_id),
        )
        con.execute(
            "INSERT INTO players (job_id, track_id, official_id, team) VALUES (?,?,?,?)",
            (job_id, track_id, official_id, team),
        )


def get_players(job_id: str) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM players WHERE job_id = ? ORDER BY team, official_id",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_job(job_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM players WHERE job_id = ?", (job_id,))
        con.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
