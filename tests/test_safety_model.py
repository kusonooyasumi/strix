"""Tests for the dedicated safety-review model configuration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from strix.config import loader
from strix.config.settings import SafetySettings
from strix.safety.reviewer import _safety_extra_args, _safety_model_settings


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _model_settings(safety: SafetySettings, model_name: str) -> object:
    """The reviewer's per-call settings, with the main model's headers as fallback."""
    return _safety_model_settings(safety, model_name, loader.load_settings().llm.extra_headers)


def test_safety_key_sent_per_call_not_via_global_env() -> None:
    safety = SafetySettings(STRIX_SAFETY_MODEL="deepseek/cheap", SAFETY_LLM_API_KEY="safety-key")
    # The key rides on the request, so a shared-provider main key can't clobber
    # it (and vice versa) through the global provider env var.
    assert _safety_extra_args(safety)["api_key"] == "safety-key"


def test_safety_extra_args_empty_without_a_dedicated_model() -> None:
    # A key with no dedicated model would leak onto the main model's requests.
    safety = SafetySettings(SAFETY_LLM_API_KEY="safety-key")
    assert _safety_extra_args(safety) == {}


def test_safety_settings_omit_credentials_when_unset() -> None:
    safety = SafetySettings(STRIX_SAFETY_MODEL="deepseek/cheap")
    extra = _safety_extra_args(safety)
    assert "api_key" not in extra
    assert "api_base" not in extra


def test_safety_endpoint_sent_per_call() -> None:
    safety = SafetySettings(
        STRIX_SAFETY_MODEL="openai/cheap",
        SAFETY_LLM_API_KEY="safety-key",
        SAFETY_LLM_API_BASE="https://safety.example/v1",
    )
    extra = _safety_extra_args(safety)
    # A distinct safety endpoint rides on the request instead of the
    # process-wide base URL, so it can't clobber the main model's endpoint.
    assert extra["api_base"] == "https://safety.example/v1"
    assert extra["api_key"] == "safety-key"


def test_dedicated_safety_model_uses_own_headers_not_main() -> None:
    safety = SafetySettings(
        STRIX_SAFETY_MODEL="deepseek/cheap",
        SAFETY_LLM_EXTRA_HEADERS={"X-Safety": "yes"},
    )
    settings = _model_settings(safety, "deepseek/cheap")
    assert settings.extra_headers == {"X-Safety": "yes"}


def test_dedicated_safety_model_gets_no_main_headers_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_EXTRA_HEADERS", json.dumps({"X-Main": "secret"}))
    loader._cached = None
    try:
        safety = SafetySettings(STRIX_SAFETY_MODEL="deepseek/cheap")
        settings = _model_settings(safety, "deepseek/cheap")
        assert settings.extra_headers is None
    finally:
        loader._cached = None


def test_fallback_safety_inherits_main_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_EXTRA_HEADERS", json.dumps({"X-Main": "svc"}))
    loader._cached = None
    try:
        settings = _model_settings(SafetySettings(), "openai/main-model")
        assert settings.extra_headers == {"X-Main": "svc"}
    finally:
        loader._cached = None


def test_safety_defaults_leave_the_model_unset() -> None:
    settings = SafetySettings()
    assert settings.model is None
    assert settings.api_key is None
    assert settings.api_base is None
    assert settings.extra_headers is None


def test_safety_model_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIX_SAFETY_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("SAFETY_LLM_API_KEY", "k")

    settings = SafetySettings()

    assert settings.model == "deepseek/deepseek-v4-flash"
    assert settings.api_key == "k"


def test_config_file_loads_safety_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("STRIX_LLM", "STRIX_SAFETY_MODEL", "SAFETY_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "env": {
                    "STRIX_LLM": "openai/root",
                    "STRIX_SAFETY_MODEL": "deepseek/cheap",
                    "SAFETY_LLM_API_KEY": "safety-key",
                }
            }
        ),
        encoding="utf-8",
    )
    loader._cached = None
    loader._override = path
    try:
        settings = loader.load_settings()
    finally:
        loader._cached = None
        loader._override = None

    assert settings.safety.model == "deepseek/cheap"
    assert settings.safety.api_key == "safety-key"
    # Main model stays independent of the safety override.
    assert settings.llm.model == "openai/root"
