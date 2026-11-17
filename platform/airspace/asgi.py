"""ASGI entry point: uvicorn airspace.asgi:app"""
from .app import create_app

app = create_app()
