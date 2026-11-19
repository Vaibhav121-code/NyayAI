
import sqlite3
from pathlib import Path
from typing import Any


class IngestionTracker:
    """Track PDF ingestion state using SQLite."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    error TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def get(self, path: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT path, content_hash, status, chunk_count, error
                FROM documents WHERE path = ?
                """,
                (path,),
            ).fetchone()

        if row is None:
            return None

        return {
            "path": row[0],
            "content_hash": row[1],
            "status": row[2],
            "chunk_count": row[3],
            "error": row[4],
        }

    @classmethod
    def read_all(cls, db_path: str | Path) -> dict[str, dict[str, Any]]:
        """Read the tracker snapshot without creating or changing the database."""
        database_path = Path(db_path)
        if not database_path.exists():
            return {}

        with sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True, timeout=30) as connection:
            rows = connection.execute(
                "SELECT path, content_hash, status, chunk_count, error FROM documents"
            ).fetchall()

        return {
            row[0]: {
                "path": row[0],
                "content_hash": row[1],
                "status": row[2],
                "chunk_count": row[3],
                "error": row[4],
            }
            for row in rows
        }

    def mark_processing(self, path: str, content_hash: str):
        self._save(path, content_hash, "processing", 0, None)

    def mark_completed(
        self,
        path: str,
        content_hash: str,
        chunk_count: int,
    ):
        self._save(path, content_hash, "completed", chunk_count, None)

    def mark_failed(
        self,
        path: str,
        content_hash: str,
        error: str,
    ):
        self._save(path, content_hash, "failed", 0, error)

    def _save(
        self,
        path: str,
        content_hash: str,
        status: str,
        chunk_count: int,
        error: str | None,
    ):
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (path, content_hash, status, chunk_count, error, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    status = excluded.status,
                    chunk_count = excluded.chunk_count,
                    error = excluded.error,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (path, content_hash, status, chunk_count, error),
            )

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) FROM documents GROUP BY status"
            ).fetchall()

        return {status: count for status, count in rows}
