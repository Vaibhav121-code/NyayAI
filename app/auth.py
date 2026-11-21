from __future__ import annotations

from contextlib import contextmanager
import hashlib
import hmac
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from collections.abc import Iterator
from typing import Any

SESSION_COOKIE_NAME = "nyay_session"
SESSION_TTL_SECONDS = 7 * 24 * 60 * 60
SCRYPT_N = 1 << 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_KEY_LENGTH = 64
SCRYPT_MAX_MEMORY = 64 * 1024 * 1024

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class InvalidEmailError(ValueError):
    pass


class DuplicateEmailError(ValueError):
    pass


def normalize_email(email: str) -> str:
    normalized = email.strip().casefold()
    if len(normalized) > 254 or not _EMAIL_PATTERN.fullmatch(normalized):
        raise InvalidEmailError("Enter a valid email address.")
    return normalized


def _database_path() -> Path:
    configured_path = os.getenv("AUTH_DB_PATH")
    if configured_path:
        return Path(configured_path)
    return Path(__file__).resolve().parents[1] / "data" / "auth.sqlite3"


@contextmanager
def _database() -> Iterator[sqlite3.Connection]:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_user_id_idx ON sessions(user_id);
            """
        )
        with connection:
            yield connection
    finally:
        connection.close()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_KEY_LENGTH,
        maxmem=SCRYPT_MAX_MEMORY,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded_hash: str) -> bool:
    parts = encoded_hash.split("$")
    if len(parts) != 6 or parts[:4] != [
        "scrypt",
        str(SCRYPT_N),
        str(SCRYPT_R),
        str(SCRYPT_P),
    ]:
        return False

    try:
        salt = bytes.fromhex(parts[4])
        expected_digest = bytes.fromhex(parts[5])
    except ValueError:
        return False
    if len(salt) != 16 or len(expected_digest) != SCRYPT_KEY_LENGTH:
        return False

    actual_digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_KEY_LENGTH,
        maxmem=SCRYPT_MAX_MEMORY,
    )
    return hmac.compare_digest(actual_digest, expected_digest)


def _token_digest(token: str) -> str:
    try:
        encoded_token = token.encode("ascii")
    except UnicodeEncodeError:
        return ""
    return hashlib.sha256(encoded_token).hexdigest()


def _new_session(connection: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    connection.execute(
        "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
        (_token_digest(token), user_id, int(time.time()) + SESSION_TTL_SECONDS),
    )
    return token


def create_account(email: str, password: str) -> tuple[dict[str, Any], str]:
    password_hash = hash_password(password)
    try:
        with _database() as connection:
            cursor = connection.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email, password_hash, int(time.time())),
            )
            user = {"id": int(cursor.lastrowid), "email": email}
            token = _new_session(connection, user["id"])
            return user, token
    except sqlite3.IntegrityError as exc:
        if "users.email" in str(exc):
            raise DuplicateEmailError("An account with this email already exists.") from exc
        raise


def login(email: str, password: str) -> tuple[dict[str, Any], str] | None:
    with _database() as connection:
        row = connection.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None

        user = {"id": int(row["id"]), "email": str(row["email"])}
        token = _new_session(connection, user["id"])
        return user, token


def user_for_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None

    with _database() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.email, sessions.expires_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
            """,
            (_token_digest(token),),
        ).fetchone()
        if row is None:
            return None
        if int(row["expires_at"]) <= int(time.time()):
            connection.execute(
                "DELETE FROM sessions WHERE token_hash = ?",
                (_token_digest(token),),
            )
            return None
        return {"id": int(row["id"]), "email": str(row["email"])}


def delete_session(token: str | None) -> None:
    if not token:
        return
    with _database() as connection:
        connection.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (_token_digest(token),),
        )
