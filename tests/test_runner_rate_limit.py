"""Tests for graceful handling of persistent provider errors in run_strix_scan.

A persistent rate limit or a quota/billing exhaustion is a resumable pause, not a
scan bug, so the root agent settles to ``stopped`` and the run returns ``None``
with a resume hint. A genuine (non-quota) API error still fails the scan.
"""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from openai import APIError, RateLimitError

import strix.tools.notes.tools as notes_tools
import strix.tools.todo.tools as todo_tools
from strix.core import runner
from strix.core.agents import AgentCoordinator


if TYPE_CHECKING:
    from collections.abc import Callable


def _make_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError("rate limited", response=response, body=None)


def _make_quota_error() -> APIError:
    """A quota/billing error as it surfaces mid-stream: a bare ``APIError`` with a
    provider error code but no HTTP status code."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APIError(
        message="You exceeded your current quota, please check your plan and billing details.",
        request=request,
        body={"type": "insufficient_quota", "code": "insufficient_quota"},
    )


def _make_bad_request_error() -> APIError:
    """A genuine, non-resumable API error (should still fail the scan)."""
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return APIError(
        message="Invalid value for 'tools': too many tools.",
        request=request,
        body={"type": "invalid_request_error", "code": "invalid_request_error"},
    )


async def _run_scan_with_agent_loop_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    error_factory: Callable[[], BaseException],
) -> tuple[Any, AgentCoordinator]:
    """Drive ``run_strix_scan`` with a ``run_agent_loop`` that always raises."""
    monkeypatch.setattr(runner, "run_dir_for", lambda _scan_id: tmp_path)
    monkeypatch.setattr(runner, "runtime_state_dir", lambda _run_dir: tmp_path)
    monkeypatch.setattr(runner, "setup_scan_logging", lambda _run_dir: lambda: None)
    monkeypatch.setattr(runner, "set_scan_id", lambda _scan_id: None)

    settings = types.SimpleNamespace(
        llm=types.SimpleNamespace(
            model="openai/gpt-4o",
            reasoning_effort="high",
            force_required_tool_choice=False,
        ),
        runtime=types.SimpleNamespace(max_context_images=3),
    )
    monkeypatch.setattr(runner, "load_settings", lambda: settings)
    monkeypatch.setattr(runner, "configure_sdk_model_defaults", lambda _settings: None)
    monkeypatch.setattr(
        runner, "uses_chat_completions_tool_schema", lambda _model, _settings: False
    )

    monkeypatch.setattr(todo_tools, "hydrate_todos_from_disk", lambda _state_dir: None)
    monkeypatch.setattr(notes_tools, "hydrate_notes_from_disk", lambda _state_dir: None)

    async def _create_or_reuse(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"client": object(), "session": object(), "caido_client": None}

    async def _cleanup(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(runner.session_manager, "create_or_reuse", _create_or_reuse)  # type: ignore[attr-defined]
    monkeypatch.setattr(runner.session_manager, "cleanup", _cleanup)  # type: ignore[attr-defined]

    monkeypatch.setattr(runner, "build_root_task", lambda _scan_config: "task")
    monkeypatch.setattr(runner, "build_scope_context", lambda _scan_config: "")
    monkeypatch.setattr(runner, "make_model_settings", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runner, "build_strix_agent", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "make_child_factory", lambda **_kwargs: lambda **_k: object())
    monkeypatch.setattr(runner, "open_agent_session", lambda _root_id, _db: object())

    async def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise error_factory()

    monkeypatch.setattr(runner, "run_agent_loop", _raise)

    coordinator = AgentCoordinator()
    result = await runner.run_strix_scan(
        scan_config={"targets": [], "scan_mode": "deep"},
        scan_id="scan-test",
        image="img",
        coordinator=coordinator,
    )
    return result, coordinator


def _root_id(coordinator: AgentCoordinator) -> str:
    root_ids = [aid for aid, parent in coordinator.parent_of.items() if parent is None]
    assert len(root_ids) == 1
    return root_ids[0]


@pytest.mark.asyncio
async def test_persistent_rate_limit_stops_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A persistent RateLimitError stops the scan (root -> 'stopped') without raising."""
    with caplog.at_level(logging.WARNING):
        result, coordinator = await _run_scan_with_agent_loop_error(
            monkeypatch, tmp_path, _make_rate_limit_error
        )

    assert result is None
    assert coordinator.statuses[_root_id(coordinator)] == "stopped"
    # the resume hint must carry the real scan id, not a literal placeholder
    assert "strix --resume scan-test" in caplog.text
    assert "<run_name>" not in caplog.text


@pytest.mark.asyncio
async def test_quota_exhaustion_stops_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A mid-stream quota/billing APIError (no HTTP status) is treated as a resumable
    pause, not a hard failure: root -> 'stopped', no raise, resume hint logged."""
    with caplog.at_level(logging.WARNING):
        result, coordinator = await _run_scan_with_agent_loop_error(
            monkeypatch, tmp_path, _make_quota_error
        )

    assert result is None
    assert coordinator.statuses[_root_id(coordinator)] == "stopped"
    assert "strix --resume scan-test" in caplog.text


@pytest.mark.asyncio
async def test_non_quota_api_error_still_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A genuine (non-quota) APIError is not swallowed: it re-raises and the root
    agent is marked 'failed'."""
    with pytest.raises(APIError):
        await _run_scan_with_agent_loop_error(monkeypatch, tmp_path, _make_bad_request_error)


def test_is_resumable_provider_error_classification() -> None:
    assert runner._is_resumable_provider_error(_make_quota_error()) is True
    assert runner._is_resumable_provider_error(_make_bad_request_error()) is False
