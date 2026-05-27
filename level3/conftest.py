
import os
import pytest

def pytest_configure(config):
    # """Verify GROQ_API_KEY is set before any test runs."""
    if not os.environ.get("GROQ_API_KEY"):
        pytest.exit("GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key", returncode=1)