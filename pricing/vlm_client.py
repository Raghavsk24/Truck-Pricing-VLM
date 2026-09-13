"""Optional Cursor SDK VLM client (requires CURSOR_API_KEY).

The default pricing labeling path does NOT use this module.
Prefer the in-chat agent workflow:

  python pricing/label_sample.py          # stage + empty batches
  # Cursor agent fills pricing/batches/*.json
  python pricing/merge_batch_labels.py

Keep this file only if you later want scripted API labeling.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = os.environ.get("CURSOR_MODEL", "claude-sonnet-5-thinking-high")


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty VLM response")
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_vlm_cursor(
    staged_image: Path,
    schema: dict,
    *,
    model: str = DEFAULT_MODEL,
    cwd: Path | None = None,
    api_key: str | None = None,
) -> dict:
    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions, SDKImage, UserMessage
    except ImportError as exc:
        raise SystemExit("Install cursor-sdk: pip install cursor-sdk") from exc

    key = api_key or os.environ.get("CURSOR_API_KEY")
    if not key:
        raise SystemExit(
            "CURSOR_API_KEY not set. For no-API labeling use:\n"
            "  python pricing/label_sample.py\n"
            "  # fill batches in Cursor chat\n"
            "  python pricing/merge_batch_labels.py"
        )

    schema_text = json.dumps(schema, ensure_ascii=False)
    prompt = (
        "You are a vision labeler for Class 7/8 truck photos.\n"
        "Inspect the attached image and return ONE JSON object that matches the "
        "JSON schema below exactly.\n"
        "Output JSON only. No markdown fences, no commentary.\n\n"
        f"JSON schema:\n{schema_text}"
    )
    result = Agent.prompt(
        UserMessage(
            text=prompt,
            images=[SDKImage.from_file(str(staged_image))],
        ),
        AgentOptions(
            model=model,
            api_key=key,
            tools=[],
            local=LocalAgentOptions(cwd=str(cwd or REPO)),
        ),
    )
    if getattr(result, "status", None) == "error":
        raise RuntimeError(f"Cursor agent run failed: {getattr(result, 'id', '?')}")
    return _parse_json_object(getattr(result, "result", None) or "")
