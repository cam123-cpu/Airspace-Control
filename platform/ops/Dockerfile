# Multi-stage build: dependencies compiled in a builder, runtime image has no compilers, no shell tools, runs as non-root.
FROM python:3.13-slim AS build
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
COPY airspace ./airspace

FROM gcr.io/distroless/python3-debian12:nonroot
# NOTE: pin the distroless image by digest in production (image@sha256:...).
WORKDIR /app
COPY --from=build /install /usr/local
COPY --from=build /build/airspace /app/airspace
ENV PYTHONPATH=/usr/local/lib/python3.13/site-packages:/app \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AIRSPACE_ENV=production AIRSPACE_DB_PATH=/data/airspace.db AIRSPACE_EVIDENCE_DIR=/data/evidence
USER nonroot
EXPOSE 8080
# no Server/Date banners. uvicorn proxy-header rewriting is OFF: the app is the single place that interprets
# X-Forwarded-For (only from AIRSPACE_TRUSTED_PROXIES peers, rightmost hop) -- pentest F2
ENTRYPOINT ["python3", "-m", "uvicorn", "airspace.asgi:app", "--host", "0.0.0.0", "--port", "8080", \
            "--no-server-header", "--no-date-header", "--no-proxy-headers", \
            "--timeout-keep-alive", "5", "--limit-concurrency", "512"]
