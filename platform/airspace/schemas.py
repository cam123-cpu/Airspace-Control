"""Strict, bounded input schemas. Unknown fields are rejected; every string, number and list has limits."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Id = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{6,64}$")]
ExtId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,40}$")]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Note = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
Email = Annotated[str, StringConstraints(pattern=r"^[^@\s]{1,64}@[^@\s]{1,189}$", max_length=254)]
Coord = Annotated[float, Field(ge=-50_000, le=50_000, allow_inf_nan=False)]
Epoch = Annotated[int, Field(ge=1_600_000_000, le=4_102_444_800)]


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ChallengeIn(Strict):
    kind: Literal["pow", "turnstile", "hcaptcha"]
    token: Annotated[str, StringConstraints(min_length=1, max_length=2048)]
    solution: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{1,32}$")] | None = None


class LoginIn(Strict):
    email: Email
    password: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    totp: Annotated[str, StringConstraints(pattern=r"^[0-9]{6}$")] | None = None
    challenge: ChallengeIn | None = None


class RefreshIn(Strict):
    refresh_token: Annotated[str, StringConstraints(pattern=r"^rt_[A-Za-z0-9_-]{20,100}$")]


class ResolveIn(Strict):
    note: Note


class SensorHealthIn(Strict):
    sensor_id: Id
    status: Literal["ok", "degraded", "offline"]


class TrackIn(Strict):
    ext_track_id: ExtId
    cls: Literal["drone", "bird", "aircraft", "vehicle", "unknown"]
    confidence: Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
    x: Coord
    y: Coord
    z: Annotated[float, Field(ge=-500, le=10_000, allow_inf_nan=False)]
    observed_at: Epoch


class AlertIn(Strict):
    ext_track_id: ExtId
    severity: Literal["suspect", "alert", "spoof"]
    title: ShortText
    remote_id_text: Annotated[str, StringConstraints(max_length=200)] | None = None   # attacker-controlled over the air


class TelemetryIn(Strict):
    sensors: Annotated[list[SensorHealthIn], Field(max_length=64)] = []
    tracks: Annotated[list[TrackIn], Field(max_length=500)] = []
    alerts: Annotated[list[AlertIn], Field(max_length=50)] = []


class ApprovalIn(Strict):
    action: Literal["disable_sensor", "enable_sensor", "set_alert_threshold", "set_retention", "export_evidence"]
    sensor_id: Id | None = None
    site_id: Id | None = None
    evidence_id: Id | None = None
    threshold: Annotated[float, Field(ge=0.5, le=0.99, allow_inf_nan=False)] | None = None
    retention_days: Annotated[int, Field(ge=30, le=3650)] | None = None
    reason: Note


class UserIn(Strict):
    email: Email
    role: Literal["viewer", "operator", "supervisor", "tenant_admin"]
    password: Annotated[str, StringConstraints(min_length=12, max_length=256)]


class DeviceIn(Strict):
    site_id: Id
    public_key: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{43}$")]   # raw Ed25519 key, base64url, no padding


class ApiTokenIn(Strict):
    name: ShortText
    role: Literal["viewer", "operator"]
    allowed_tools: Annotated[list[Literal["list_active_alerts", "get_alert", "site_sensor_health", "search_tracks", "acknowledge_alert"]],
                             Field(min_length=1, max_length=5)]
    ttl_hours: Annotated[int, Field(ge=1, le=24 * 90)]
