"""Shared pytest fixtures for the LocalMind backend test suite.

The whole suite runs without Ollama and without touching the real database:
this module repoints ``LOCALMIND_DB`` at a throwaway file *before* anything
imports the logger, and provides a fixture that mocks all four model clients so
no test ever reaches ``localhost:11434``.
"""

import os
import tempfile

# Must run before any import that constructs the logger singleton.
_TEST_DB = os.path.join(tempfile.gettempdir(), "localmind_test.db")
os.environ["LOCALMIND_DB"] = _TEST_DB
if os.path.exists(_TEST_DB):
    os.remove(_TEST_DB)

from unittest.mock import AsyncMock  # noqa: E402

import pytest  # noqa: E402

import router  # noqa: E402
from models import (  # noqa: E402
    deepseek_client,
    llama32_client,
    llava_client,
    mistral_client,
)

# The canonical fake every mocked client returns when not overridden.
FAKE_RESPONSE = {"response": "mock response", "latency_ms": 100, "model": "mock"}


@pytest.fixture
def mock_clients(monkeypatch):
    """Mock all four clients' ``generate`` so tests run without Ollama.

    Returns a dict of the four ``AsyncMock`` objects (keyed ``llama32``,
    ``mistral``, ``deepseek``, ``llava``) so a test can assert call counts or
    override a single client's behaviour (e.g. raise to test error isolation).
    """
    mocks = {}
    for key, module in (
        ("llama32", llama32_client),
        ("mistral", mistral_client),
        ("deepseek", deepseek_client),
        ("llava", llava_client),
    ):
        mock = AsyncMock(return_value=dict(FAKE_RESPONSE))
        monkeypatch.setattr(module, "generate", mock)
        mocks[key] = mock

    # router captured each client's generate() in a dispatch dict at import time,
    # so patching the module attribute alone doesn't reach the text experts —
    # repoint the dict at the mocks too.
    monkeypatch.setattr(
        router,
        "_TEXT_EXPERT_CLIENTS",
        {
            "llama3.2": mocks["llama32"],
            "mistral": mocks["mistral"],
            "deepseek-r1:7b": mocks["deepseek"],
        },
    )
    return mocks
