"""Deny-by-default role-based access control. A permission not listed here is granted to nobody."""
from __future__ import annotations

from .auth import Principal

ROLES = ("viewer", "operator", "supervisor", "tenant_admin", "platform_auditor")

_ALL_TENANT = {"viewer", "operator", "supervisor", "tenant_admin"}
_OPS = {"operator", "supervisor", "tenant_admin"}
_SUP = {"supervisor", "tenant_admin"}

POLICY: dict[str, frozenset[str]] = {
    "sites.read": frozenset(_ALL_TENANT),
    "sensors.read": frozenset(_ALL_TENANT),
    "alerts.read": frozenset(_ALL_TENANT),
    "tracks.read": frozenset(_ALL_TENANT),
    "evidence.read": frozenset(_OPS),
    "alerts.ack": frozenset(_OPS),
    "alerts.resolve": frozenset(_OPS),
    "alerts.mcp_confirm": frozenset(_OPS),
    "alerts.legal_hold": frozenset({"tenant_admin"}),
    # sensitive actions: only ever executed through a two-person approval
    "approvals.request": frozenset(_SUP),
    "approvals.approve": frozenset(_SUP),
    "approvals.read": frozenset(_SUP),
    "users.manage": frozenset({"tenant_admin"}),
    "devices.manage": frozenset({"tenant_admin"}),
    "api_tokens.manage": frozenset({"tenant_admin"}),
    "audit.read": frozenset({"tenant_admin", "platform_auditor"}),
    "audit.verify": frozenset({"tenant_admin", "platform_auditor"}),
}

# Actions that are never directly callable; they run only when a second, different supervisor approves.
TWO_PERSON_ACTIONS = ("disable_sensor", "enable_sensor", "set_alert_threshold", "set_retention", "export_evidence")


class Forbidden(Exception):
    pass


def allowed(principal: Principal, permission: str) -> bool:
    roles = POLICY.get(permission)
    return bool(roles) and principal.role in roles


def require(principal: Principal, permission: str) -> None:
    if not allowed(principal, permission):
        raise Forbidden(permission)
    if principal.role != "platform_auditor" and not principal.tenant_id:
        raise Forbidden("no tenant")
