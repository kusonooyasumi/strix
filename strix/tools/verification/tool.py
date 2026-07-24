"""``submit_verification_verdict`` — verifier-agent termination + verdict capture."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import RunContextWrapper, function_tool


logger = logging.getLogger(__name__)


def _do_submit(
    *,
    confirmed: bool,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    if not reason.strip():
        return {
            "success": False,
            "error": "reason cannot be empty; explain the evidence for your verdict",
        }
    try:
        bounded = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        bounded = 0.0
    return {
        "success": True,
        "verification_completed": True,
        "confirmed": bool(confirmed),
        "confidence": bounded,
        "reason": reason.strip()[:2000],
    }


@function_tool(strict_mode=False)
async def submit_verification_verdict(
    ctx: RunContextWrapper,
    confirmed: bool,
    confidence: float,
    reason: str,
) -> str:
    """Submit your final verification verdict and stop.

    Call this once you have independently attempted to reproduce the candidate
    finding using the sandbox tools (shell, HTTP proxy, browser). Base the
    verdict only on evidence you actually observed, not on the candidate's
    claims.

    Args:
        confirmed: ``true`` only if your own reproduction demonstrated the
            claimed vulnerability and impact. ``false`` for false positives,
            unproven exploitability, contradicted evidence, or anything you
            could not reproduce.
        confidence: Your confidence in the verdict, 0.0-1.0.
        reason: Concrete, evidence-based rationale citing what you ran and
            observed (requests, responses, command output, callbacks).
    """
    _ = ctx
    result = _do_submit(confirmed=confirmed, confidence=confidence, reason=reason)
    return json.dumps(result, ensure_ascii=False, default=str)
