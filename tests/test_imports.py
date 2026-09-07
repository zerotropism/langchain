"""Smoke tests: every module in src/ must import cleanly."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

MODULES = [
    "agent",
    "config",
    "decorators",
    "history",
    "llm",
    "main",
    "memory",
    "parsing",
    "processing",
    "prompting",
    "rag",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    __import__(name)
