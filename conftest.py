# tests/conftest.py
import os
import pytest
from dotenv import load_dotenv

load_dotenv()  # allow local runs to pick up .env

REQUIRED_ENV_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
]

@pytest.fixture(scope="session")
def bedrock_env_ready():
    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        pytest.skip(f"Skipping Bedrock tests; missing env vars: {', '.join(missing)}")
    return True

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom section to pytest output summary."""
    results = terminalreporter.config.cache.get("evaluation_results", [])
    if results:
        terminalreporter.section("LLM Judge Results")
        for item in results:
            query, scores = item
            terminalreporter.write_line(f"Query: {query}")
            terminalreporter.write_line(f"  {scores}")