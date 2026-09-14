# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for :class:`MuseGlimmerRenderer` and the ATEM tool-call parser.

Runs against the REAL Muse Glimmer checkpoint tokenizer. Designed to work
both under pytest and as a plain script (``python test_renderer.py``); the
tokenizer path can be overridden via ``$GLIMMER_CKPT``.

Covered:
  1. Simple user->assistant render frames correctly (verified control ids).
  2. Tool-spec + assistant tool-call round-trips through parse_response
     (name + arguments recovered, status OK).
  3. get_stop_token_ids() includes <|eot|> + <|end_of_text|> but NOT <|eom|>.
  4. register() makes the renderer resolvable via the library registry.
  5. bridge_to_next_turn() is prefix-invariant over prev prompt+completion.
"""

from __future__ import annotations

import os
import sys

try:
    import pytest
except ModuleNotFoundError:  # plain-script mode without pytest installed
    class _PytestShim:
        """Minimal stand-in so the module imports and the ``@fixture``
        decorators work when pytest is not installed. Fixtures degrade to
        plain functions; ``_run_as_script`` wires dependencies manually."""

        @staticmethod
        def fixture(*args, **kwargs):
            def _decorator(fn):
                return fn

            # Support both @pytest.fixture and @pytest.fixture(scope=...).
            if args and callable(args[0]):
                return args[0]
            return _decorator

    pytest = _PytestShim()  # type: ignore[assignment]

# Allow running as a plain script from anywhere in the repo.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from renderers.base import ToolCallParseStatus  # noqa: E402

from torchtitan.experiments.rl.examples.glimmer_search_r1 import atem  # noqa: E402
from torchtitan.experiments.rl.examples.glimmer_search_r1.renderer import (  # noqa: E402
    MuseGlimmerRenderer,
    MuseGlimmerRendererConfig,
    register,
)

CKPT = os.environ.get(
    "GLIMMER_CKPT", "/home/alisol/projects/muse-glimmer-test/hf_ckpt"
)

# Verified control-token ids (from encoding literals against the real
# checkpoint tokenizer).
BOS = 200000
EOS = 200001
EOM = 200007
EOT = 200008
START = 200022
MESSAGE = 200023

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)


@pytest.fixture(scope="module")
def renderer(tokenizer):
    return MuseGlimmerRenderer(tokenizer)


# ── 1. control-token ids + framing ────────────────────────────────────────
def test_control_token_ids(renderer):
    assert renderer._bos == BOS
    assert renderer._eos == EOS
    assert renderer._eom == EOM
    assert renderer._eot == EOT
    assert renderer._start == START
    assert renderer._message == MESSAGE


def test_simple_render_frames(renderer):
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    rt = renderer.render(msgs, add_generation_prompt=True)

    # Parallel-array invariant.
    n = len(rt.token_ids)
    assert len(rt.message_indices) == n
    assert len(rt.sampled_mask) == n
    assert len(rt.is_content) == n

    ids = rt.token_ids
    # Starts with BOS, then the system frame opener.
    assert ids[0] == BOS
    assert ids[1] == START
    # A <|message|> and a <|eot|> appear (system + user frames).
    assert MESSAGE in ids
    assert EOT in ids
    # Ends with a bare <|start|>assistant generation prompt (no <|message|>
    # after the final <|start|>).
    assert ids[-1] != MESSAGE
    last_start = max(i for i, t in enumerate(ids) if t == START)
    assert MESSAGE not in ids[last_start:]
    # The generation prompt is pure scaffolding (never sampled).
    assert not any(rt.sampled_mask)


def test_user_content_is_body(renderer, tokenizer):
    msgs = [{"role": "user", "content": "UNIQUEMARKER123"}]
    rt = renderer.render(msgs, add_generation_prompt=True)
    body_ids = [t for t, c in zip(rt.token_ids, rt.is_content) if c]
    decoded = tokenizer.decode(body_ids, skip_special_tokens=False)
    assert "UNIQUEMARKER123" in decoded


# ── 2. tool round-trip ─────────────────────────────────────────────────────
def test_tool_call_round_trip(renderer):
    prompt_msgs = [{"role": "user", "content": "Find cats"}]
    prompt_ids = renderer.render_ids(
        prompt_msgs, tools=TOOLS, add_generation_prompt=True
    )

    full_conv = prompt_msgs + [
        {
            "role": "assistant",
            "reasoning_content": "I will search for cats.",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "cats", "limit": 5},
                    },
                }
            ],
        }
    ]
    full = renderer.render(full_conv, tools=TOOLS)
    completion_ids = full.token_ids[len(prompt_ids):]

    parsed = renderer.parse_response(completion_ids, tools=TOOLS)
    assert parsed.reasoning_content == "I will search for cats."
    ok = [tc for tc in parsed.tool_calls if tc.status == ToolCallParseStatus.OK]
    assert len(ok) == 1
    assert ok[0].name == "web_search"
    assert ok[0].arguments == {"query": "cats", "limit": 5}


def test_atem_module_round_trip():
    block = atem.render_tool_call(
        "grep", {"pattern": "foo", "flags": ["-i", "-n"], "recursive": True}
    )
    assert "<atem:invoke name=\"grep\">" in block
    calls = atem.parse_tool_calls(block)
    assert len(calls) == 1
    assert calls[0].status == atem.AtemParseStatus.OK
    assert calls[0].name == "grep"
    assert calls[0].arguments == {
        "pattern": "foo",
        "flags": ["-i", "-n"],
        "recursive": True,
    }


def test_atem_missing_name_status():
    calls = atem.parse_tool_calls(
        "<atem:invoke>\n<atem:parameter name=\"x\">1</atem:parameter>\n</atem:invoke>"
    )
    assert len(calls) == 1
    assert calls[0].status == atem.AtemParseStatus.MISSING_NAME


def test_atem_unclosed_block_status():
    calls = atem.parse_tool_calls('<atem:invoke name="foo">\n<atem:parameter name="a">1')
    assert len(calls) == 1
    assert calls[0].status == atem.AtemParseStatus.UNCLOSED_BLOCK
    assert calls[0].name == "foo"


def test_atem_invalid_json_status():
    calls = atem.parse_tool_calls(
        '<atem:invoke name="f">\n<atem:parameter name="a">{bad json</atem:parameter>\n</atem:invoke>'
    )
    assert len(calls) == 1
    assert calls[0].status == atem.AtemParseStatus.INVALID_JSON


# ── 3. stop tokens ─────────────────────────────────────────────────────────
def test_stop_token_ids(renderer):
    stops = renderer.get_stop_token_ids()
    # <|eot|> (turn end) and <|end_of_text|> are hard stops.
    assert EOT in stops
    assert renderer._eos in stops
    # <|eom|> (intra-turn message end) must NOT be a stop: it terminates the
    # reasoning frame, and stopping there prevents the model from emitting the
    # subsequent tool-call / final-answer frame (verified on the 30B ckpt —
    # empty completions -> zero reward variance -> no trainable RL batch).
    assert EOM not in stops


# ── 4. registry resolution ─────────────────────────────────────────────────
def test_register_resolvable(tokenizer):
    import renderers.base as rbase
    import renderers.configs as rconfigs
    from renderers import create_renderer

    register()

    assert rbase.RENDERER_REGISTRY.get("muse_glimmer") is MuseGlimmerRenderer
    assert rbase.RENDERER_REGISTRY.get("glimmer") is MuseGlimmerRenderer
    assert (
        rconfigs._CONFIG_BY_NAME.get("muse_glimmer") is MuseGlimmerRendererConfig
    )
    assert rconfigs._config_class_for("glimmer") is MuseGlimmerRendererConfig
    assert (
        rbase.MODEL_RENDERER_MAP.get("meta-models/Muse-Glimmer-30B")
        == "muse_glimmer"
    )

    # Resolvable via create_renderer with the typed config.
    r = create_renderer(tokenizer, MuseGlimmerRendererConfig())
    assert isinstance(r, MuseGlimmerRenderer)

    # Resolvable via config_from_name (the string-name path).
    cfg = rconfigs.config_from_name("muse_glimmer")
    assert isinstance(cfg, MuseGlimmerRendererConfig)


def test_register_idempotent():
    register()
    register()
    import renderers.base as rbase

    assert rbase.RENDERER_REGISTRY.get("muse_glimmer") is MuseGlimmerRenderer


# ── 5. bridge prefix-invariance ────────────────────────────────────────────
def test_bridge_prefix_invariance(renderer):
    # Turn 1: user -> assistant tool call.
    prompt_msgs = [{"role": "user", "content": "Find cats"}]
    prompt_ids = renderer.render_ids(
        prompt_msgs, tools=TOOLS, add_generation_prompt=True
    )
    full_conv = prompt_msgs + [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "cats"},
                    },
                }
            ],
        }
    ]
    full = renderer.render(full_conv, tools=TOOLS)
    completion_ids = full.token_ids[len(prompt_ids):]
    assert len(completion_ids) > 0

    # Turn 2: a tool result arrives.
    new_messages = [
        {"role": "tool", "name": "web_search", "content": "Cats are felines."}
    ]
    bridged = renderer.bridge_to_next_turn(
        prompt_ids, completion_ids, new_messages, tools=TOOLS
    )
    assert bridged is not None

    prev = list(prompt_ids) + list(completion_ids)
    # Prefix-invariance: bridged.token_ids begins with prev_prompt+prev_completion.
    assert bridged.token_ids[: len(prev)] == prev

    # Bridge ends at a fresh assistant generation prompt (bare <|start|>assistant).
    assert bridged.token_ids[-1] != MESSAGE
    last_start = max(i for i, t in enumerate(bridged.token_ids) if t == START)
    assert MESSAGE not in bridged.token_ids[last_start:]

    # Bridge attribution invariants.
    n = len(bridged.token_ids)
    assert len(bridged.message_indices) == n
    assert len(bridged.sampled_mask) == n
    assert len(bridged.is_content) == n
    # sampled_mask uniformly False on a bridge result.
    assert not any(bridged.sampled_mask)
    # Prior portion is attributed to -1.
    assert all(mi == -1 for mi in bridged.message_indices[: len(prev)])


def test_bridge_rejects_assistant(renderer):
    prompt_ids = renderer.render_ids(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True
    )
    completion_ids = renderer.render_ids(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    )[len(prompt_ids):]
    out = renderer.bridge_to_next_turn(
        prompt_ids,
        completion_ids,
        [{"role": "assistant", "content": "no"}],
    )
    assert out is None


def _run_as_script():
    """Run every test in this module without pytest."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
    r = MuseGlimmerRenderer(tok)

    test_control_token_ids(r)
    test_simple_render_frames(r)
    test_user_content_is_body(r, tok)
    test_tool_call_round_trip(r)
    test_atem_module_round_trip()
    test_atem_missing_name_status()
    test_atem_unclosed_block_status()
    test_atem_invalid_json_status()
    test_stop_token_ids(r)
    test_register_resolvable(tok)
    test_register_idempotent()
    test_bridge_prefix_invariance(r)
    test_bridge_rejects_assistant(r)
    print("ALL TESTS PASSED (script mode)")


if __name__ == "__main__":
    _run_as_script()
