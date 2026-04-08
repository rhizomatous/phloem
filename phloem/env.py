"""load .env file from project root."""

from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)
