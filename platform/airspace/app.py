"""HTTP API (Starlette). Routes are thin: authenticate -> parse/validate -> call the shared service layer."""
from __future__ import annotations
import ipaddress

import json
import re
from typing import Awaitable, Callable

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

import hmac

from . import admin, alerts, approvals, audit, challenge, crypto, evidence, ingest, net, sessions
from .auth import AuthError, Principal, principal_from_access_token
from .config import Settings, load_settings
from .db import DB, migrate
from .errors import AppError, ForbiddenErr, Invalid, TooMany, Unauthorized
from .outputs import trim
from .security_headers import jlog
from .ratelimit import check
from .rbac import Forbidden, require
from .security_headers import HardeningMiddleware, json_depth_ok

_ID = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


RT_COOKIE = "__Host-airspace_rt"        # refresh token: HttpOnly, never readable by page script
CSRF_COOKIE = "__Host-airspace_csrf"    # double-submit token: readable by the console, echoed in X-CSRF-Token


def create_app(settings: Settings | None = None) -> Starlette:
    s = settings or load_settings()
    migrate(s.db_path)
    crypto.install(s.keyring())

    def client_ip(req: Request) -> str:
        peer = req.client.host if req.client else "unknown"
        return net.client_ip(s, peer, {k.lower(): v for k, v in req.headers.items()})

    def out(name: str, data, status: int = 200, headers: dict | None = None) -> JSONResponse:
        body, dropped = trim(name, data, s.strict_outputs)
        if dropped:
            jlog(level="warn", event="response_fields_dropped", route=name, fields=dropped[:10])
        return JSONResponse(body, status_code=status, headers=headers)

    async def body_json(req: Request) -> dict:
        if req.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise Invalid("content-type must be application/json")
        raw = await req.body()
        try:
            data = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys)
        except (ValueError, RecursionError):
            raise Invalid("malformed JSON")
        if not isinstance(data, dict) or not json_depth_ok(data, s.max_json_depth):
            raise Invalid("JSON must be an object of limited depth")
        return data

    def path_id(req: Request, name: str) -> str:
        v = req.path_params[name]
        if not _ID.match(v):
            raise Invalid("bad id")
        return v

    def authed(handler: Callable[[Request, DB, Principal], Awaitable[Response]]):
        async def wrapper(req: Request) -> Response:
            db = DB(s.db_path)
            try:
                auth = req.headers.get("authorization", "")
                if not auth.startswith("Bearer ") or len(auth) > 4096:
                    raise Unauthorized()
                try:
                    p = principal_from_access_token(s, db, auth[7:].strip())
                except AuthError:
                    raise Unauthorized()
                if not check("api_user", p.id):
                    raise TooMany()
                return await handler(req, db, p)
            except Forbidden:
                return _err(ForbiddenErr())
            except ValidationError:
                return _err(Invalid("input failed validation"))
            except AppError as e:
                return _err(e)
            finally:
                db.close()
        return wrapper

    def public(handler: Callable[[Request, DB], Awaitable[Response]]):
        async def wrapper(req: Request) -> Response:
            db = DB(s.db_path)
            try:
                return await handler(req, db)
            except ValidationError:
                return _err(Invalid("input failed validation"))
            except AppError as e:
                return _err(e)
            finally:
                db.close()
        return wrapper

    # ---------- auth ----------
    def set_session_cookies(resp: JSONResponse, rt: str | None) -> JSONResponse:
        if rt is None:
            for name in (RT_COOKIE, CSRF_COOKIE):
                resp.set_cookie(name, "", max_age=0, path="/", secure=True, httponly=(name == RT_COOKIE), samesite="strict")
            return resp
        resp.set_cookie(RT_COOKIE, rt, max_age=s.refresh_ttl_s, path="/", secure=True, httponly=True, samesite="strict")
        resp.set_cookie(CSRF_COOKIE, sessions.csrf_token_for(s, rt), max_age=s.refresh_ttl_s, path="/", secure=True,
                        httponly=False, samesite="strict")
        return resp

    def require_allowed_origin(req: Request) -> None:
        """Cookie sessions only: the Origin header must be present and in the allowlist (blocks cross-site use)."""
        origin = req.headers.get("origin", "")
        if not origin or origin not in s.cors_origins:
            raise ForbiddenErr("origin not allowed")

    def cookie_refresh_token(req: Request) -> str:
        """Read the refresh cookie and enforce CSRF: allowed Origin + X-CSRF-Token equal to the CSRF cookie and to
        the value bound to this refresh token."""
        require_allowed_origin(req)
        rt = req.cookies.get(RT_COOKIE, "")
        sent = req.headers.get("x-csrf-token", "")
        cookie = req.cookies.get(CSRF_COOKIE, "")
        if not rt or not sent or not cookie:
            raise Unauthorized()
        expected = sessions.csrf_token_for(s, rt)
        if not (hmac.compare_digest(sent, cookie) and hmac.compare_digest(sent, expected)):
            raise ForbiddenErr("csrf check failed")
        return rt

    def cookie_mode(req: Request) -> bool:
        return req.headers.get("x-session-mode", "").lower() == "cookie"

    @public
    async def auth_challenge(req, db):
        return out("challenge", challenge.issue_pow(s, client_ip(req)))

    @public
    async def login(req, db):
        body = await body_json(req)
        if cookie_mode(req):
            require_allowed_origin(req)            # login CSRF: a foreign site can't plant a session
        res = sessions.login(s, db, body, client_ip(req))
        if cookie_mode(req):
            rt = res.pop("refresh_token")
            res["csrf_token"] = sessions.csrf_token_for(s, rt)
            return set_session_cookies(out("login", res), rt)
        return out("login", res)

    @public
    async def refresh(req, db):
        body = await body_json(req)
        if cookie_mode(req):
            if body:
                raise Invalid("cookie sessions send an empty JSON body")
            res = sessions.refresh(s, db, {"refresh_token": cookie_refresh_token(req)}, client_ip(req))
            rt = res.pop("refresh_token")
            res["csrf_token"] = sessions.csrf_token_for(s, rt)
            return set_session_cookies(out("refresh", res), rt)
        return out("refresh", sessions.refresh(s, db, body, client_ip(req)))

    @public
    async def logout(req, db):
        body = await body_json(req)
        if cookie_mode(req):
            if body:
                raise Invalid("cookie sessions send an empty JSON body")
            res = sessions.logout(db, {"refresh_token": cookie_refresh_token(req)})
            return set_session_cookies(out("logout", res), None)
        return out("logout", sessions.logout(db, body))

    @authed
    async def me(req, db, p):
        return out("me", {"id": p.id, "tenant_id": p.tenant_id, "role": p.role})

    # ---------- read ----------
    @authed
    async def sites(req, db, p):
        return out("sites", {"items": alerts.list_sites(db, p)})

    @authed
    async def sensors(req, db, p):
        return out("sensors", {"items": alerts.site_sensors(db, p, path_id(req, "site_id"))})

    @authed
    async def alert_list(req, db, p):
        qp = req.query_params
        return out("alert_list", alerts.list_alerts(db, p, qp.get("status"), qp.get("site_id") if qp.get("site_id") and _ID.match(qp.get("site_id")) else None,
                                                    qp.get("limit", 50), qp.get("cursor")))

    @authed
    async def alert_get(req, db, p):
        return out("alert_get", alerts.get_alert(db, p, path_id(req, "alert_id")))

    @authed
    async def tracks(req, db, p):
        qp = req.query_params
        site = qp.get("site_id", "")
        if not _ID.match(site):
            raise Invalid("site_id required")
        return out("tracks", {"items": alerts.list_tracks(db, p, site, qp.get("since", 0), qp.get("limit", 100))})

    # ---------- alert actions ----------
    @authed
    async def alert_ack(req, db, p):
        return out("alert_ack", alerts.acknowledge(db, p, path_id(req, "alert_id")))

    @authed
    async def alert_resolve(req, db, p):
        from .schemas import ResolveIn
        b = ResolveIn.model_validate(await body_json(req))
        return out("alert_resolve", alerts.resolve(db, p, path_id(req, "alert_id"), b.note))

    @authed
    async def alert_hold(req, db, p):
        b = await body_json(req)
        if set(b) != {"hold"} or not isinstance(b["hold"], bool):
            raise Invalid("expected {\"hold\": true|false}")
        return out("alert_hold", alerts.set_legal_hold(db, p, path_id(req, "alert_id"), b["hold"]))

    @authed
    async def alert_mcp_confirm(req, db, p):
        return out("mcp_confirmation", alerts.new_mcp_confirmation(db, p, path_id(req, "alert_id")))

    # ---------- evidence ----------
    @authed
    async def ev_meta(req, db, p):
        return out("evidence_meta", evidence.metadata(s, db, p, path_id(req, "evidence_id")))

    @authed
    async def ev_download(req, db, p):
        data, ctype, fname = evidence.download(s, db, p, path_id(req, "evidence_id"), req.headers.get("x-export-grant", ""))
        return Response(data, media_type=ctype, headers={"content-disposition": f'attachment; filename="{fname}"',
                                                          "x-content-type-options": "nosniff"})

    # ---------- two-person approvals ----------
    @authed
    async def appr_create(req, db, p):
        return out("approval_create", approvals.request(s, db, p, await body_json(req)), 201)

    @authed
    async def appr_list(req, db, p):
        return out("approval_list", {"items": approvals.list_pending(db, p)})

    @authed
    async def appr_approve(req, db, p):
        return out("approval_decide", approvals.decide(s, db, p, path_id(req, "approval_id"), True))

    @authed
    async def appr_reject(req, db, p):
        return out("approval_decide", approvals.decide(s, db, p, path_id(req, "approval_id"), False))

    # ---------- admin ----------
    @authed
    async def users_create(req, db, p):
        return out("user_create", admin.create_user(s, db, p, await body_json(req)), 201)

    @authed
    async def devices_create(req, db, p):
        return out("device_create", admin.register_device(db, p, await body_json(req)), 201)

    @authed
    async def tokens_create(req, db, p):
        return out("token_create", admin.create_api_token(db, p, await body_json(req)), 201)

    @authed
    async def tokens_revoke(req, db, p):
        return out("token_revoke", admin.revoke_api_token(db, p, path_id(req, "token_id")))

    # ---------- audit ----------
    @authed
    async def audit_list(req, db, p):
        require(p, "audit.read")
        tid = _audit_tenant(req, p)
        try:
            after = int(req.query_params.get("after", 0))
        except ValueError:
            raise Invalid("bad after")
        rows = db.q("SELECT seq, ts, actor, action, target, detail, hash FROM audit_log WHERE tenant_id = ? AND seq > ? ORDER BY seq LIMIT 200", (tid, after))
        return out("audit_list", {"items": rows})

    @authed
    async def audit_verify(req, db, p):
        require(p, "audit.verify")
        return out("audit_verify", audit.verify(db, _audit_tenant(req, p)))

    def _audit_tenant(req: Request, p: Principal) -> str:
        if p.role == "platform_auditor":   # cross-tenant by design, read-only
            tid = req.query_params.get("tenant_id", "")
            if not re.match(r"^(tn_[0-9a-f]{16}|_platform)$", tid):
                raise Invalid("tenant_id required")
            return tid
        return p.tenant_id or ""

    # ---------- device ingest (signed) ----------
    async def ingest_route(req: Request) -> Response:
        db = DB(s.db_path)
        try:
            raw = await req.body()
            hdrs = {k.lower(): v for k, v in req.headers.items()}
            dev, site = ingest.verify_device(s, db, req.method, req.url.path + ("?" + req.url.query if req.url.query else ""), hdrs, raw)
            if req.url.path == "/ingest/v1/telemetry":
                try:
                    data = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_no_duplicate_keys)
                except (ValueError, RecursionError):
                    raise Invalid("malformed JSON")
                if not isinstance(data, dict) or not json_depth_ok(data, s.max_json_depth):
                    raise Invalid("bad JSON")
                return out("ingest_telemetry", ingest.ingest_telemetry(db, dev, site, data, req.headers.get("idempotency-key")), 202)
            alert_id = req.query_params.get("alert_id", "")
            if not _ID.match(alert_id):
                raise Invalid("alert_id required")
            return out("ingest_evidence", evidence.store(s, db, dev, site, alert_id, raw), 201)
        except ValidationError:
            return _err(Invalid("input failed validation"))
        except AppError as e:
            return _err(e)
        finally:
            db.close()

    async def healthz(req: Request) -> Response:
        return out("healthz", {"ok": True})

    routes = [
        Route("/healthz", healthz, methods=["GET"]),
        Route("/api/v1/auth/challenge", auth_challenge, methods=["POST"]),
        Route("/api/v1/auth/login", login, methods=["POST"]),
        Route("/api/v1/auth/refresh", refresh, methods=["POST"]),
        Route("/api/v1/auth/logout", logout, methods=["POST"]),
        Route("/api/v1/me", me, methods=["GET"]),
        Route("/api/v1/sites", sites, methods=["GET"]),
        Route("/api/v1/sites/{site_id}/sensors", sensors, methods=["GET"]),
        Route("/api/v1/alerts", alert_list, methods=["GET"]),
        Route("/api/v1/alerts/{alert_id}", alert_get, methods=["GET"]),
        Route("/api/v1/alerts/{alert_id}/ack", alert_ack, methods=["POST"]),
        Route("/api/v1/alerts/{alert_id}/resolve", alert_resolve, methods=["POST"]),
        Route("/api/v1/alerts/{alert_id}/legal-hold", alert_hold, methods=["POST"]),
        Route("/api/v1/alerts/{alert_id}/mcp-confirmation", alert_mcp_confirm, methods=["POST"]),
        Route("/api/v1/tracks", tracks, methods=["GET"]),
        Route("/api/v1/evidence/{evidence_id}", ev_meta, methods=["GET"]),
        Route("/api/v1/evidence/{evidence_id}/download", ev_download, methods=["GET"]),
        Route("/api/v1/approvals", appr_create, methods=["POST"]),
        Route("/api/v1/approvals", appr_list, methods=["GET"]),
        Route("/api/v1/approvals/{approval_id}/approve", appr_approve, methods=["POST"]),
        Route("/api/v1/approvals/{approval_id}/reject", appr_reject, methods=["POST"]),
        Route("/api/v1/users", users_create, methods=["POST"]),
        Route("/api/v1/devices", devices_create, methods=["POST"]),
        Route("/api/v1/api-tokens", tokens_create, methods=["POST"]),
        Route("/api/v1/api-tokens/{token_id}/revoke", tokens_revoke, methods=["POST"]),
        Route("/api/v1/audit", audit_list, methods=["GET"]),
        Route("/api/v1/audit/verify", audit_verify, methods=["GET"]),
        Route("/ingest/v1/telemetry", ingest_route, methods=["POST"]),
        Route("/ingest/v1/evidence", ingest_route, methods=["POST"]),
    ]

    app = Starlette(routes=routes, exception_handlers={404: _route_404, 405: _route_405})
    app.add_middleware(HardeningMiddleware, settings=s)
    app.state.settings = s
    return app


async def _route_404(req, exc):
    return JSONResponse({"error": "not_found"}, status_code=404)


async def _route_405(req, exc):
    return JSONResponse({"error": "method_not_allowed"}, status_code=405)


def _reject_constant(c: str):
    raise ValueError("NaN/Infinity not allowed")


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """A repeated key means two parsers can read two different requests (a proxy that keeps the first value,
    this app that would keep the last). Refuse rather than pick one."""
    d: dict = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate key")
        d[k] = v
    return d


def _err(e: AppError) -> JSONResponse:
    body = {"error": e.code, "message": e.message if e.status < 500 else e.code}
    if "challenge" in e.extra:
        body["challenge"] = e.extra["challenge"]
    return JSONResponse(trim("error", body)[0], status_code=e.status)
