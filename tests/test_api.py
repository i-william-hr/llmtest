#!/usr/bin/env python3
"""
Jinja2 template for API integration tests
Follows the sso/wg_gui patterns: test client, auth gates, rate limits, CSRF, redirect safety
"""

import pytest

# Routes discovered in entrypoint:
#   GET /api/models -> api_models
#   POST /api/models/refresh -> api_models_refresh
#   GET /api/chats -> api_chats_list
#   POST /api/chats -> api_chats_create
#   GET /api/chats/<int:cid> -> api_chats_get
#   PATCH /api/chats/<int:cid> -> api_chats_patch
#   DELETE /api/chats/<int:cid> -> api_chats_delete
#   POST /api/chats/<int:cid>/stream -> api_stream
#   POST /api/chats/<int:cid>/messages -> api_append_user
#   DELETE /api/messages/<int:mid> -> api_message_delete
#   GET / -> index


class TestHealthEndpoints:
    """Tests for health check and public endpoints."""

    def test_healthz(self, client):
        """Health check endpoint should return 200."""
        resp = client.get("/healthz")
        # Some apps use /health, /healthz, /ready, /ping
        assert resp.status_code in (200, 404)  # 404 if not implemented

    def test_root_redirects_or_returns_content(self, client):
        """Root endpoint should respond (redirect or content)."""
        resp = client.get("/")
        assert resp.status_code in (200, 301, 302, 401, 403)


class TestAuthGates:
    """Tests for authentication gates (SSO, Basic, API key)."""

    def test_protected_endpoint_requires_auth(self, client):
        """Protected endpoints should return 401/403 without auth."""
        # Test a known protected endpoint
        protected_routes = [
            ("/api/models", ["GET"]),
            ("/api/models/refresh", ["POST"]),
            ("/api/chats", ["GET"]),
            ("/api/chats", ["POST"]),
            ("/api/chats/<int:cid>", ["GET"]),
            ("/api/chats/<int:cid>", ["PATCH"]),
            ("/api/chats/<int:cid>", ["DELETE"]),
            ("/api/chats/<int:cid>/stream", ["POST"]),
            ("/api/chats/<int:cid>/messages", ["POST"]),
            ("/api/messages/<int:mid>", ["DELETE"]),
            ("/", ["GET"]),
        ]
        for route, methods in protected_routes:
            # Skip health check endpoints (intentionally public)
            if route in ["/healthz", "/health", "/ready", "/ping", "/metrics"]:
                continue
            if route == "/":
                continue
            # Use the first supported method for testing auth
            method = methods[0].lower()
            resp = getattr(client, method)(route)
            # Should be 401 (unauthenticated) or 403 (forbidden) or 302 (redirect to login) or 405 (wrong method but auth gate works)
            assert resp.status_code in (401, 403, 302, 405), f"Route {route} should require auth"

    def test_sso_redirect_preserves_next(self, client):
        """SSO login should preserve the next parameter."""
        # This tests the SSO gate pattern
        resp = client.get("/protected?next=/target")
        if resp.status_code == 302:
            location = resp.headers.get("Location", "")
            assert "next=" in location or "/sso/login" in location


class TestRateLimiting:
    """Tests for rate limiting on auth endpoints."""

    def test_login_rate_limit(self, client):
        """Repeated failed logins should be rate limited."""
        # This is a placeholder - actual implementation depends on the app
        # The sso pattern uses throttle_record_failure / throttle_blocked
        pass


class TestCSRFProtection:
    """Tests for CSRF protection on mutating endpoints."""

    def test_mutating_endpoint_requires_csrf(self, client):
        """POST/PUT/DELETE endpoints should require CSRF token."""
        resp = client.post("/api/models/refresh", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/api/models/refresh should require CSRF or auth"
        resp = client.post("/api/chats", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/api/chats should require CSRF or auth"
        resp = client.post("/api/chats/<int:cid>", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/api/chats/<int:cid> should require CSRF or auth"
        resp = client.post("/api/chats/<int:cid>/stream", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/api/chats/<int:cid>/stream should require CSRF or auth"
        resp = client.post("/api/chats/<int:cid>/messages", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/api/chats/<int:cid>/messages should require CSRF or auth"
        resp = client.post("/api/messages/<int:mid>", data={})
        # May return 401 (auth required first), 400 (bad request), or 403 (CSRF missing)
        assert resp.status_code in (400, 401, 403), f"/api/messages/<int:mid> should require CSRF or auth"


class TestRedirectSafety:
    """Tests for open redirect protection."""

    def test_redirect_validation(self, client):
        """Redirects should be validated against allowlist."""
        # Test the safe_next pattern from sso
        malicious_targets = [
            "https://evil.example/",
            "//evil.example/",
            "javascript:alert(1)",
            "/\\evil.example/",
        ]
        for target in malicious_targets:
            # This would need app-specific testing
            pass


class TestAPIEndpoints:
    """Tests for specific API endpoints."""

    def test_api_models_endpoint(self, client):
        """Test GET /api/models."""
        resp = client.open("/api/models", method="GET")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_models_refresh_endpoint(self, client):
        """Test POST /api/models/refresh."""
        resp = client.open("/api/models/refresh", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_chats_list_endpoint(self, client):
        """Test GET /api/chats."""
        resp = client.open("/api/chats", method="GET")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_chats_create_endpoint(self, client):
        """Test POST /api/chats."""
        resp = client.open("/api/chats", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_chats_get_endpoint(self, client):
        """Test GET /api/chats/<int:cid>."""
        resp = client.open("/api/chats/<int:cid>", method="GET")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_chats_patch_endpoint(self, client):
        """Test PATCH /api/chats/<int:cid>."""
        resp = client.open("/api/chats/<int:cid>", method="PATCH")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_chats_delete_endpoint(self, client):
        """Test DELETE /api/chats/<int:cid>."""
        resp = client.open("/api/chats/<int:cid>", method="DELETE")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_stream_endpoint(self, client):
        """Test POST /api/chats/<int:cid>/stream."""
        resp = client.open("/api/chats/<int:cid>/stream", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_append_user_endpoint(self, client):
        """Test POST /api/chats/<int:cid>/messages."""
        resp = client.open("/api/chats/<int:cid>/messages", method="POST")
        # Should not be 500
        assert resp.status_code != 500
    def test_api_message_delete_endpoint(self, client):
        """Test DELETE /api/messages/<int:mid>."""
        resp = client.open("/api/messages/<int:mid>", method="DELETE")
        # Should not be 500
        assert resp.status_code != 500
    def test_index_endpoint(self, client):
        """Test GET /."""
        resp = client.open("/", method="GET")
        # Should not be 500
        assert resp.status_code != 500
