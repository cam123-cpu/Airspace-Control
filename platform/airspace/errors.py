"""Domain errors mapped to HTTP status codes by the app. Messages are short and safe to show."""


class AppError(Exception):
    status = 400
    code = "bad_request"

    def __init__(self, message: str = "", extra: dict | None = None):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.extra = extra or {}


class NotFound(AppError):
    status, code = 404, "not_found"      # also used for other tenants' objects: existence is never revealed


class Conflict(AppError):
    status, code = 409, "conflict"


class Unauthorized(AppError):
    status, code = 401, "unauthorized"


class ForbiddenErr(AppError):
    status, code = 403, "forbidden"


class TooMany(AppError):
    status, code = 429, "rate_limited"


class ChallengeRequired(AppError):
    status, code = 428, "challenge_required"    # body carries a fresh challenge to solve


class HttpsRequired(AppError):
    status, code = 400, "https_required"


class Invalid(AppError):
    status, code = 422, "invalid_input"
