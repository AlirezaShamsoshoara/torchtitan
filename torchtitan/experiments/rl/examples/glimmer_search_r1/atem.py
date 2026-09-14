# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ATEM tool-call format for Muse Glimmer.

ATEM ("agentic tool-execution markup") is Muse Glimmer's XML-ish
function-call block. It is distinct from the JSON ``tool_calls`` wire
format used by e.g. Qwen3/Hermes: arguments are emitted as named
``<atem:parameter>`` elements, not a single JSON object.

Wire format (VERIFIED against the real checkpoint's ``response_template``
in ``tokenizer_config.json`` and the reference Jinja chat template shipped
with the checkpoint family — see the module docstring in ``renderer.py``
for provenance)::

    <atem:function_calls>
    <atem:invoke name="TOOL_NAME">
    <atem:parameter name="KEY_1">VALUE_1</atem:parameter>
    <atem:parameter name="KEY_2">VALUE_2</atem:parameter>
    </atem:invoke>
    </atem:function_calls>

Value serialization (matching the reference template's ``render_atem``):
  * ``bool``      -> ``true`` / ``false``
  * ``None``      -> ``null``
  * ``dict`` / non-string iterable -> compact JSON (``json.dumps``)
  * everything else (str, int, float) -> the value verbatim (``str(v)``)

Parsing mirrors the checkpoint's declared parser
(``value_parser={"name": "json", "args": {"allow_non_json": True}}``):
each parameter value is first tried as JSON; on failure the raw string is
kept verbatim. That "allow_non_json" fallback is why scalars can be
emitted bare on the wire and still round-trip.

RECONCILE-BEFORE-PRODUCTION NOTE
--------------------------------
The exact ATEM wire format here was reconstructed from the checkpoint's
``response_template`` grammar and the reference chat template. It should
be reconciled against the official Muse Glimmer renderer/spec upstream
before production use. The parser and renderer are deliberately isolated
in this module (a single ``render_tool_call`` / ``parse_tool_calls`` pair
plus the delimiter constants) so that swapping in the canonical wire
format — should it differ — is a localized change that does not touch the
token-level bookkeeping in ``renderer.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Wire-format delimiters (single source of truth) ───────────────────────
FUNCTION_CALLS_OPEN = "<atem:function_calls>"
FUNCTION_CALLS_CLOSE = "</atem:function_calls>"
INVOKE_CLOSE = "</atem:invoke>"
PARAMETER_CLOSE = "</atem:parameter>"


class AtemParseStatus(str, Enum):
    """Per-invoke parse outcome, mirroring ``renderers.ToolCallParseStatus``.

    Kept as an independent enum so ``atem.py`` has no import dependency on
    the ``renderers`` package (which drags in ``transformers``); the
    renderer maps these onto ``renderers.ToolCallParseStatus`` 1:1.
    """

    OK = "ok"
    INVALID_JSON = "invalid_json"
    MISSING_NAME = "missing_name"
    UNCLOSED_BLOCK = "unclosed_block"
    MALFORMED_STRUCTURE = "malformed_structure"


@dataclass
class AtemToolCall:
    """One parsed ``<atem:invoke>`` attempt.

    ``raw`` is the block text as emitted; ``name`` / ``arguments`` are the
    recovered function name and argument dict (``arguments`` may hold raw
    strings for values that failed JSON parsing under the allow-non-json
    fallback). ``status`` distinguishes clean vs malformed attempts.
    """

    raw: str
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    status: AtemParseStatus = AtemParseStatus.OK


# ── Rendering ──────────────────────────────────────────────────────────────
def _render_value(v: Any) -> str:
    """Serialize a single argument value for the ATEM wire format.

    Matches the reference chat template's ``render_atem`` value handling:
    booleans lowercase, ``None`` -> ``null``, containers -> compact JSON,
    scalars verbatim.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, str):
        return v
    if isinstance(v, dict) or (
        hasattr(v, "__iter__") and not isinstance(v, (str, bytes))
    ):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def render_tool_call(name: str, arguments: Any) -> str:
    """Render a single tool call as an ATEM ``<atem:function_calls>`` block.

    ``arguments`` may be a dict (preferred) or a JSON string; a JSON string
    is decoded first so the parameters expand into ``<atem:parameter>``
    elements. A non-dict, non-JSON string is wrapped under a single
    ``value`` parameter so nothing is silently dropped.

    Note: the reference template wraps *each* tool call in its own
    ``<atem:function_calls>`` block (one invoke per block). This function
    renders exactly one such block; the renderer emits one assistant
    message per tool call, matching the checkpoint's per-call framing.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    parts = [FUNCTION_CALLS_OPEN, "\n", '<atem:invoke name="', name, '">', "\n"]
    for k, v in arguments.items():
        parts.append('<atem:parameter name="')
        parts.append(str(k))
        parts.append('">')
        parts.append(_render_value(v))
        parts.append(PARAMETER_CLOSE)
        parts.append("\n")
    parts.append(INVOKE_CLOSE)
    parts.append("\n")
    parts.append(FUNCTION_CALLS_CLOSE)
    return "".join(parts)


# ── Parsing ──────────────────────────────────────────────────────────────
# Verified against the checkpoint response_template's regex grammar:
#   open_pattern:  <atem:invoke\b[^>]*?\bname="(?P<name>[^"]+)">
#   tag_pattern:   <atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>
_INVOKE_OPEN_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="(?P<name>[^"]*)"[^>]*?>')
_INVOKE_OPEN_NONAME_RE = re.compile(r"<atem:invoke\b[^>]*?>")
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)


def _parse_value(raw: str) -> tuple[Any, bool]:
    """Parse one parameter value under the allow-non-json fallback.

    Returns ``(value, ok)`` where ``ok`` is ``False`` only when the value
    looked like it was meant to be JSON (starts with a JSON structural
    char) but failed to parse — that surfaces as ``INVALID_JSON``. A bare
    scalar string that isn't JSON is kept verbatim with ``ok=True``,
    matching ``allow_non_json=True``.
    """
    stripped = raw.strip()
    try:
        return json.loads(stripped), True
    except (json.JSONDecodeError, ValueError):
        # allow_non_json: keep the verbatim string. Only flag as a JSON
        # error when the value clearly intended structured JSON.
        if stripped[:1] in "{[":
            return raw, False
        return raw, True


def parse_tool_calls(text: str) -> list[AtemToolCall]:
    """Parse every ``<atem:invoke>`` attempt in ``text``.

    Robust to the surrounding ``<atem:function_calls>`` wrapper being
    present or absent — the model's sampled stream is scanned for invoke
    blocks directly. Returns one :class:`AtemToolCall` per attempt, in
    order, successful and malformed alike (status distinguishes them),
    mirroring the ``renderers`` parser contract.

    Status semantics:
      * ``OK``               — invoke opened, closed, name present, all
                               parameter values parsed (or kept verbatim).
      * ``UNCLOSED_BLOCK``   — an ``<atem:invoke ...>`` opener with no
                               matching ``</atem:invoke>`` (truncated /
                               hit stop token).
      * ``MISSING_NAME``     — invoke block with no ``name="..."``.
      * ``INVALID_JSON``     — a parameter value intended as JSON failed
                               to parse.
      * ``MALFORMED_STRUCTURE`` — a stray ``</atem:invoke>`` close with no
                               opener, etc.
    """
    calls: list[AtemToolCall] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _INVOKE_OPEN_RE.search(text, pos)
        m_noname = _INVOKE_OPEN_NONAME_RE.search(text, pos)
        # Prefer whichever opener comes first; a no-name opener that
        # precedes a named one is a MISSING_NAME attempt.
        if m is None and m_noname is None:
            break
        if m is not None and (m_noname is None or m.start() <= m_noname.start()):
            opener = m
            name: str | None = m.group("name")
        else:
            opener = m_noname
            name = None

        close_idx = text.find(INVOKE_CLOSE, opener.end())
        if close_idx == -1:
            # Unclosed invoke — everything to end of text is the raw block.
            raw = text[opener.start():]
            calls.append(
                AtemToolCall(
                    raw=raw,
                    name=name if name else None,
                    arguments={},
                    status=AtemParseStatus.UNCLOSED_BLOCK
                    if name
                    else AtemParseStatus.MISSING_NAME,
                )
            )
            break

        body_start = opener.end()
        raw = text[opener.start(): close_idx + len(INVOKE_CLOSE)]
        inner = text[body_start:close_idx]

        if not name:
            calls.append(
                AtemToolCall(
                    raw=raw, name=None, arguments={},
                    status=AtemParseStatus.MISSING_NAME,
                )
            )
            pos = close_idx + len(INVOKE_CLOSE)
            continue

        arguments: dict[str, Any] = {}
        status = AtemParseStatus.OK
        for pm in _PARAM_RE.finditer(inner):
            key = pm.group("key")
            val, ok = _parse_value(pm.group("value"))
            arguments[key] = val
            if not ok:
                status = AtemParseStatus.INVALID_JSON
        calls.append(
            AtemToolCall(raw=raw, name=name, arguments=arguments, status=status)
        )
        pos = close_idx + len(INVOKE_CLOSE)

    return calls


__all__ = [
    "FUNCTION_CALLS_OPEN",
    "FUNCTION_CALLS_CLOSE",
    "INVOKE_CLOSE",
    "PARAMETER_CLOSE",
    "AtemParseStatus",
    "AtemToolCall",
    "render_tool_call",
    "parse_tool_calls",
]
