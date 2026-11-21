import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def auth_client(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_DB_PATH", str(tmp_path / "auth.sqlite3"))
    with TestClient(main.app) as client:
        yield client


def test_root_shows_auth_page_and_search_requires_login(auth_client):
    page = auth_client.get("/")
    protected = auth_client.post("/api/query", json={"query": "Article 21"})

    assert page.status_code == 200
    assert 'id="authForm"' in page.text
    assert protected.status_code == 401
    assert protected.json()["detail"] == "Please log in to continue."


def test_root_is_not_cached_across_login_and_logout(auth_client):
    signed_out_page = auth_client.get("/")
    assert signed_out_page.headers["cache-control"] == "no-store"
    assert signed_out_page.headers["vary"] == "Cookie"
    assert 'id="authForm"' in signed_out_page.text

    signup = auth_client.post(
        "/api/auth/signup",
        json={"email": "user@example.com", "password": "safe-password-123"},
    )
    assert signup.status_code == 201

    signed_in_page = auth_client.get("/")
    assert signed_in_page.headers["cache-control"] == "no-store"
    assert 'id="accountEmail"' in signed_in_page.text

    logout = auth_client.post("/api/auth/logout")
    assert logout.status_code == 200

    signed_out_again_page = auth_client.get("/")
    assert 'id="authForm"' in signed_out_again_page.text


def test_all_research_endpoints_require_login(auth_client):
    responses = [
        auth_client.get("/api/status"),
        auth_client.post("/api/retrieve", json={"query": "Article 21"}),
        auth_client.post(
            "/api/analyze-case",
            json={
                "case_summary": "Case summary.",
                "question": "Compare authorities.",
                "document_ids": ["doc-1"],
            },
        ),
        auth_client.get("/api/documents/chunk-1"),
    ]

    assert [response.status_code for response in responses] == [401, 401, 401, 401]


def test_signup_hashes_password_sets_secure_session_and_logout_revokes_it(auth_client, tmp_path):
    response = auth_client.post(
        "/api/auth/signup",
        json={"email": "User@Example.com", "password": "safe-password-123"},
    )

    assert response.status_code == 201
    assert response.json()["user"]["email"] == "user@example.com"
    cookie = response.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie

    with sqlite3.connect(tmp_path / "auth.sqlite3") as connection:
        password_hash = connection.execute(
            "SELECT password_hash FROM users WHERE email = ?",
            ("user@example.com",),
        ).fetchone()[0]
    assert password_hash.startswith("scrypt$")
    assert "safe-password-123" not in password_hash

    current_user = auth_client.get("/api/auth/me")
    assert current_user.json() == {
        "authenticated": True,
        "user": {"id": 1, "email": "user@example.com"},
    }

    logout_response = auth_client.post("/api/auth/logout")
    assert logout_response.status_code == 200
    assert logout_response.json()["authenticated"] is False
    assert auth_client.get("/api/auth/me").json()["authenticated"] is False
    assert auth_client.post("/api/query", json={"query": "Article 21"}).status_code == 401


def test_login_is_case_insensitive_and_duplicate_email_is_rejected(auth_client):
    signup = auth_client.post(
        "/api/auth/signup",
        json={"email": "user@example.com", "password": "safe-password-123"},
    )
    assert signup.status_code == 201

    duplicate = auth_client.post(
        "/api/auth/signup",
        json={"email": "USER@example.com", "password": "another-password-123"},
    )
    assert duplicate.status_code == 409

    auth_client.cookies.clear()
    invalid_login = auth_client.post(
        "/api/auth/login",
        json={"email": "user@example.com", "password": "wrong-password-123"},
    )
    assert invalid_login.status_code == 401
    assert invalid_login.json()["detail"] == "Invalid email or password."

    login = auth_client.post(
        "/api/auth/login",
        json={"email": "USER@example.com", "password": "safe-password-123"},
    )
    assert login.status_code == 200
    assert login.json()["authenticated"] is True
    assert auth_client.get("/api/auth/me").json()["user"]["email"] == "user@example.com"


def test_signup_rejects_invalid_email_and_short_password(auth_client):
    invalid_email = auth_client.post(
        "/api/auth/signup",
        json={"email": "not-an-email", "password": "safe-password-123"},
    )
    short_password = auth_client.post(
        "/api/auth/signup",
        json={"email": "user@example.com", "password": "short"},
    )

    assert invalid_email.status_code == 422
    assert invalid_email.json()["detail"] == "Enter a valid email address."
    assert short_password.status_code == 422
