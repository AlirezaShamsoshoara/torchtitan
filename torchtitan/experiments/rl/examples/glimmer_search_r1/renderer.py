# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MuseGlimmerRenderer — hand-coded renderer for the Muse Glimmer chat format.

Muse Glimmer (``meta-models/Muse-Glimmer-30B``) uses a harmony-style,
channel-based chat format in the same family as OpenAI's gpt-oss / harmony,
but with its own control-token ids and its own ATEM tool-call wire format
(see :mod:`atem`). This renderer mirrors the reference chat template shipped
with the checkpoint family and conforms to the ``renderers.Renderer``
Protocol (``render`` / ``render_ids`` / ``parse_response`` /
``get_stop_token_ids`` / ``bridge_to_next_turn``).

Provenance of the wire format (all VERIFIED against the real checkpoint at
``meta-models/Muse-Glimmer-30B`` / the local tokenizer):

  * Control-token ids confirmed by encoding the literal strings against the
    checkpoint tokenizer:
        <|begin_of_text|> = 200000   <|end_of_text|> = 200001
        <|eom|>           = 200007   <|eot|>          = 200008
        <|start|>         = 200022   <|message|>      = 200023
  * The frame grammar and the ATEM tool-call format are taken from the
    checkpoint's ``tokenizer_config.json`` ``response_template`` (which
    declares the reasoning channel ``to=self``, the content channel
    ``to=user``, the ``<|eot|>`` / ``<|eom|>`` closers, and the ATEM
    ``open_pattern`` / ``tag_pattern`` regexes) and the reference Jinja
    chat template shipped with the checkpoint family.

Per-message frame (harmony family)::

    <|start|>{header}<|message|>{body}{<|eot|> | <|eom|>}

where ``{header}`` encodes the role and, for assistant turns, the harmony
"recipient" routing:

  * user       :  ``user``
  * system     :  ``system``
  * tool       :  ``tool {name}`` with body ``<tool_output name="{name}">\n{content}\n</tool_output>``
  * assistant reasoning :  ``assistant to=self``   (closed with ``<|eom|>``)
  * assistant tool call :  ``assistant to={fn_name}`` with an ATEM body
                           (closed with ``<|eom|>`` mid-turn, ``<|eot|>`` if last)
  * assistant content   :  ``assistant to={recipient|user}``
                           (closed with ``<|eot|>`` to end the turn)

``<|eom|>`` (end-of-message) ends a non-terminal message inside a turn
(reasoning, an intermediate tool call); ``<|eot|>`` (end-of-turn) ends the
assistant's turn. The whole conversation is prefixed with a single
``<|begin_of_text|>`` (BOS); the generation prompt is a bare
``<|start|>assistant``.

The token stream is built the same way as the other hand-coded renderers in
the ``renderers`` package (``renderers.qwen3`` is the closest structural
analog): special tokens are emitted by id, body text is encoded with
``add_special_tokens=False`` (the tokenizer's ``TemplateProcessing`` would
otherwise auto-prepend BOS), and per-token ``sampled_mask`` / ``is_content``
bookkeeping follows the same "scaffold vs body vs model-sampled" contract
documented on :class:`renderers.RenderedTokens`.

RECONCILE-BEFORE-PRODUCTION: the ATEM wire format (in :mod:`atem`) was
reconstructed from the checkpoint grammar + reference template and should be
reconciled against the official Muse Glimmer renderer/spec upstream before
production. It is isolated in :mod:`atem` so a swap is localized.
"""

from __future__ import annotations

from typing import Any, Literal

from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import (
    Message,
    ParsedResponse,
    ParsedToolCall,
    RenderedTokens,
    ToolCallParseStatus,
    ToolSpec,
    attribute_text_segments,
    extract_message_tool_names,
    reject_assistant_in_extension,
    resolve_thinking_retention,
    should_rerender_for_thinking_retention,
    trim_to_turn_close,
)
from renderers.configs import BaseRendererConfig

from . import atem


# ── Config ─────────────────────────────────────────────────────────────────
class MuseGlimmerRendererConfig(BaseRendererConfig):
    """Renderer config for the Muse Glimmer chat format.

    ``reasoning_strength`` mirrors the reference template's
    ``Reasoning strength: {strength}.`` system-preamble line. It is
    renderer-internal (it shapes the synthesized system block, not a
    Jinja chat-template kwarg toggle), so it is classified as an internal
    field.

    ``use_system_prompt`` controls whether a default system block is
    synthesized when the caller supplies none but tools and/or a
    generation prompt are present (matching the reference template's
    behaviour). Also renderer-internal.
    """

    name: Literal["muse_glimmer"] = "muse_glimmer"

    reasoning_strength: str = "high"
    """Value emitted in the ``Reasoning strength: {strength}.`` preamble
    line. Mirrors the reference template's ``reasoning_strength`` kwarg."""

    use_system_prompt: bool = True
    """Synthesize a default system block (persona + reasoning strength +
    tool defs + valid recipients) when the caller supplies no system
    message but tools and/or a generation prompt are present. Matches the
    reference template."""

    _internal_fields = frozenset({"reasoning_strength", "use_system_prompt"})
    _template_fields = frozenset()


# Registry names this renderer registers under / maps models to.
_RENDERER_NAME = "muse_glimmer"
_ALIAS_NAMES = ("glimmer",)
_MODEL_NAMES = ("meta-models/Muse-Glimmer-30B",)

_DEFAULT_SYSTEM_PERSONA = "You are a helpful AI assistant."


def _content_text(content: Any) -> str:
    """Flatten content (string or list of parts) to plain text.

    Image / video parts collapse to their sentinel tokens; the text RL
    path does not exercise multimodal content, but keeping the sentinels
    avoids silently dropping structure.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text":
                out.append(p.get("text", ""))
            elif ptype == "thinking":
                out.append(p.get("thinking", ""))
            elif ptype == "image":
                out.append("<|image|>")
            elif ptype == "video":
                out.append("<|video|>")
        return "".join(out)
    return str(content)


class MuseGlimmerRenderer:
    """Deterministic message -> token renderer for Muse Glimmer (harmony)."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: MuseGlimmerRendererConfig | None = None,
    ):
        self._tokenizer = tokenizer
        self.config = config or MuseGlimmerRendererConfig()
        # No history-thinking dropping knob in this format's template, so
        # the implied bridge policy is "all" (reasoning is emitted per
        # turn and the bridge never carries assistant turns anyway).
        self.effective_thinking_retention = resolve_thinking_retention(
            self.config, "all"
        )

        self._bos = self._token_id("<|begin_of_text|>")
        self._eot = self._token_id("<|eot|>")
        self._eom = self._token_id("<|eom|>")
        self._start = self._token_id("<|start|>")
        self._message = self._token_id("<|message|>")
        self._eos = self._token_id("<|end_of_text|>")

    # ── token utilities ──────────────────────────────────────────────────
    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert (
            isinstance(tid, int)
            and tid >= 0
            and tid != self._tokenizer.unk_token_id
        ), f"Special token {token!r} not found in tokenizer vocabulary"
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        # add_special_tokens=False: the tokenizer's TemplateProcessing
        # post-processor auto-prepends <|begin_of_text|> otherwise, which
        # would corrupt every body encode. We emit BOS explicitly once.
        return self._tokenizer.encode(text, add_special_tokens=False)

    # ── system-block construction ────────────────────────────────────────
    def _tool_defs_text(self, tools: list[ToolSpec]) -> str:
        """Render the tool-definition block for the system preamble.

        Mirrors the reference template's ``render_tool_defs``: an
        instruction paragraph plus one JSON schema per function. Kept as a
        single scaffold string (never attributed as message body).
        """
        lines: list[str] = []
        lines.append(
            "In this environment you have access to a set of tools you "
            "can use to answer the user's question.\n\n"
        )
        lines.append(
            "You can invoke a function by writing a "
            '"<atem:function_calls>" block like the following:\n'
        )
        lines.append(
            "<atem:function_calls>\n"
            '<atem:invoke name="$FUNCTION_NAME">\n'
            '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE'
            "</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\n"
        )
        lines.append(
            "String and scalar parameters should be specified as is, "
            "while lists and objects should use JSON format.\n"
        )
        lines.append("Here are the functions available in JSONSchema format:\n")
        import json as _json

        for tool in tools:
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            spec = {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {},
            }
            lines.append(_json.dumps(spec, ensure_ascii=False))
            lines.append("\n")
        return "".join(lines).rstrip("\n")

    def _valid_recipients_text(self, tools: list[ToolSpec] | None) -> str:
        """The ``# Valid recipients: ...`` line from the reference template."""
        recipients = ['"self"']
        seen: list[str] = []
        for tool in tools or []:
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            ns = str(fn.get("name", "")).split(".")[0]
            if ns and ns not in seen:
                seen.append(ns)
        for ns in seen:
            recipients.append(f'"{ns}.*"')
        recipients.append('"user"')
        return "# Valid recipients: " + ", ".join(recipients) + "."

    def _system_body(
        self, persona: str, tools: list[ToolSpec] | None
    ) -> str:
        """Assemble the full system-message body (persona + preamble)."""
        parts = [persona, "\n\n", f"Reasoning strength: {self.config.reasoning_strength}."]
        if tools:
            parts.append("\n\n")
            parts.append(self._tool_defs_text(tools))
        parts.append("\n\n")
        parts.append(self._valid_recipients_text(tools))
        return "".join(parts)

    # ── public interface ─────────────────────────────────────────────────
    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if not messages:
            raise ValueError("No messages provided.")

        tokens: list[int] = []
        indices: list[int] = []
        sampled: list[bool] = []
        content_mask: list[bool] = []

        def emit_special(
            token_id: int, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)
            sampled.append(is_sampled)
            content_mask.append(is_content)

        def emit_text(
            text: str, msg_idx: int, *, is_sampled: bool, is_content: bool
        ) -> None:
            ids = self._encode(text)
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))
            sampled.extend([is_sampled] * len(ids))
            content_mask.extend([is_content] * len(ids))

        def emit_segments(
            segments: list[tuple[str, bool]], msg_idx: int, *, is_sampled: bool
        ) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                tokens.append(tok_id)
                indices.append(msg_idx)
                sampled.append(is_sampled)
                content_mask.append(is_content)

        # ── BOS ──────────────────────────────────────────────────────────
        emit_special(self._bos, -1, is_sampled=False, is_content=False)

        first_system_idx = next(
            (i for i, m in enumerate(messages) if m.get("role") == "system"),
            None,
        )

        # ── Synthesized default system block ──────────────────────────────
        # The reference template synthesizes a default system block when no
        # system message is present but tools and/or a generation prompt
        # are. It is pure scaffolding (never sampled), attributed to -1.
        if (
            first_system_idx is None
            and self.config.use_system_prompt
            and (add_generation_prompt or tools)
        ):
            emit_special(self._start, -1, is_sampled=False, is_content=False)
            emit_text("system", -1, is_sampled=False, is_content=False)
            emit_special(self._message, -1, is_sampled=False, is_content=False)
            emit_text(
                self._system_body(_DEFAULT_SYSTEM_PERSONA, tools),
                -1,
                is_sampled=False,
                is_content=False,
            )
            emit_special(self._eot, -1, is_sampled=False, is_content=False)

        # ── Iterate messages ──────────────────────────────────────────────
        n = len(messages)
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            # end token: <|eom|> if the next message shares this role,
            # else <|eot|> (matches the reference template's end_token).
            same_next = i + 1 < n and messages[i + 1].get("role") == role

            if role == "system":
                emit_special(self._start, i, is_sampled=False, is_content=False)
                emit_text("system", i, is_sampled=False, is_content=False)
                emit_special(self._message, i, is_sampled=False, is_content=False)
                # Body: caller system content is the only body-attributed
                # span; the appended preamble (reasoning strength, tool
                # defs, valid recipients) is scaffold.
                persona = _content_text(msg.get("content")) or _DEFAULT_SYSTEM_PERSONA
                caller_has_content = bool(_content_text(msg.get("content")))
                # Build the segments so the persona (caller content) is
                # is_content=True and the rest scaffold.
                tail_parts = [
                    "\n\n",
                    f"Reasoning strength: {self.config.reasoning_strength}.",
                ]
                if tools:
                    tail_parts.append("\n\n")
                    tail_parts.append(self._tool_defs_text(tools))
                tail_parts.append("\n\n")
                tail_parts.append(self._valid_recipients_text(tools))
                segs: list[tuple[str, bool]] = [(persona, caller_has_content)]
                segs.append(("".join(tail_parts), False))
                emit_segments(segs, i, is_sampled=False)
                emit_special(self._eot, i, is_sampled=False, is_content=False)

            elif role == "user":
                emit_special(self._start, i, is_sampled=False, is_content=False)
                emit_text("user", i, is_sampled=False, is_content=False)
                emit_special(self._message, i, is_sampled=False, is_content=False)
                user_segs: list[tuple[str, bool]] = []
                content = _content_text(msg.get("content"))
                if content:
                    user_segs.append((content, True))
                if user_segs:
                    emit_segments(user_segs, i, is_sampled=False)
                emit_special(self._eot, i, is_sampled=False, is_content=False)

            elif role == "tool":
                self._render_tool(
                    msg, i, emit_special=emit_special, emit_segments=emit_segments
                )

            elif role == "assistant":
                self._render_assistant(
                    msg,
                    i,
                    is_last=(i == n - 1),
                    emit_special=emit_special,
                    emit_text=emit_text,
                    emit_segments=emit_segments,
                )
            else:
                # Unknown role: emit as a system-style block, scaffold only.
                emit_special(self._start, i, is_sampled=False, is_content=False)
                emit_text(role, i, is_sampled=False, is_content=False)
                emit_special(self._message, i, is_sampled=False, is_content=False)
                emit_text(
                    _content_text(msg.get("content")),
                    i,
                    is_sampled=False,
                    is_content=True,
                )
                emit_special(
                    self._eom if same_next else self._eot,
                    i,
                    is_sampled=False,
                    is_content=False,
                )

        # ── Generation prompt: bare <|start|>assistant ────────────────────
        if add_generation_prompt:
            emit_special(self._start, -1, is_sampled=False, is_content=False)
            emit_text("assistant", -1, is_sampled=False, is_content=False)

        return RenderedTokens(
            token_ids=tokens,
            message_indices=indices,
            sampled_mask=sampled,
            is_content=content_mask,
            message_roles=[m.get("role") or "" for m in messages],
            message_tool_names=extract_message_tool_names(messages),
        )

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def parse_response(
        self,
        token_ids: list[int],
        *,
        tools: list[ToolSpec] | None = None,  # noqa: ARG002 — ATEM values are self-describing
    ) -> ParsedResponse:
        """Parse an assistant completion back into a structured message.

        The completion is the token stream the model sampled after a
        ``<|start|>assistant`` generation prompt. It may contain, in order:
          * a reasoning message  ``to=self<|message|>{reasoning}<|eom|>``
          * zero or more tool-call messages
            ``to={fn}<|message|>{ATEM}<|eom|or eot|>``
          * a final content message
            ``to=user<|message|>{content}<|eot|>``

        We decode the stripped stream to text and split on the harmony
        frame markers, routing ``to=self`` bodies to ``reasoning_content``,
        ATEM bodies to ``tool_calls`` (via :mod:`atem`), and ``to=user``
        (or unrouted) bodies to ``content``.
        """
        ids = self._strip_stop_tokens(list(token_ids))
        text = self._tokenizer.decode(ids, skip_special_tokens=False)

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: list[ParsedToolCall] = []

        for header, body in self._iter_frames(text):
            recipient = self._recipient_from_header(header)
            if recipient == "self":
                reasoning_parts.append(body)
            elif "<atem:invoke" in body or "<atem:function_calls" in body:
                for atc in atem.parse_tool_calls(body):
                    tool_calls.append(self._to_parsed_tool_call(atc))
            elif recipient in (None, "user", "") or not recipient.startswith(
                ("functions", "tool")
            ):
                content_parts.append(body)
            else:
                # A recipient that names a tool namespace but whose body
                # carries no ATEM block: still try to parse, else treat as
                # content so nothing is dropped.
                atcs = atem.parse_tool_calls(body)
                if atcs:
                    for atc in atcs:
                        tool_calls.append(self._to_parsed_tool_call(atc))
                else:
                    content_parts.append(body)

        # Fallback: no harmony frames recovered (e.g. the model emitted a
        # bare ATEM block or plain text without a leading <|start|>). Parse
        # the whole text for ATEM calls and treat the remainder as content.
        if not reasoning_parts and not content_parts and not tool_calls:
            atcs = atem.parse_tool_calls(text)
            if atcs:
                for atc in atcs:
                    tool_calls.append(self._to_parsed_tool_call(atc))
            else:
                content_parts.append(text)

        return ParsedResponse(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
        )

    def get_stop_token_ids(self) -> list[int]:
        # <|eot|> ends the assistant TURN; <|end_of_text|> is a hard stop.
        #
        # <|eom|> ends an intermediate MESSAGE within a turn (reasoning
        # `to=self`, or a tool call `to={fn}`). It must NOT be a generation
        # stop: the reference chat template concatenates
        #   reasoning<|eom|>  ->  {tool_call<|eom|> | answer<|eot|>}
        # in ONE continuous assistant turn. Stopping on <|eom|> truncates the
        # model right after its reasoning frame, so it never emits the tool
        # call or the final answer (empty `content`, no `tool_calls` -> the
        # search env ends the rollout with an empty answer -> reward 0 for
        # every sample -> zero-variance groups -> no trainable batch).
        #
        # Verified on the released 30B checkpoint: with <|eom|> as a stop the
        # model emits only `to=self` reasoning (32 tokens) and halts; with it
        # removed the model flows reasoning -> `assistant to=search` ATEM tool
        # call (parsed OK) and stops on <|eot|>. The released tokenizer has no
        # distinct tool-call terminator (only <|eom|> and <|eot|>), so the
        # tool-call boundary is recovered structurally by parse_response.
        return [self._eot, self._eos]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> RenderedTokens | None:
        """Extend prev prompt+completion with the next turn's tokens.

        Keeps the previously-sampled tokens verbatim (prefix invariance):
        the returned ``token_ids`` begin with
        ``trim_to_turn_close(prev_prompt, prev_completion)``, then append
        the rendered new (non-assistant) messages and a bare
        ``<|start|>assistant`` generation prompt. Returns ``None`` when the
        contract can't be guaranteed (empty prev, assistant in new
        messages, or an unbridgeable role).
        """
        if (
            not previous_prompt_ids
            or not new_messages
            or reject_assistant_in_extension(new_messages)
        ):
            return None

        if should_rerender_for_thinking_retention(
            self.effective_thinking_retention, new_messages
        ):
            return None

        # The prior turn closes on <|eot|> or <|eom|>. If the completion
        # was truncated at max_tokens with no close token, synthesize the
        # canonical turn close (<|eot|>).
        previous_ids = trim_to_turn_close(
            previous_prompt_ids,
            previous_completion_ids,
            {self._eot, self._eom},
            synthesize_close=self._eot,
        )
        if previous_ids is None:
            return None

        ext: list[int] = []
        ext_indices: list[int] = []
        ext_content: list[bool] = []

        def emit_special(token_id: int, msg_idx: int, *, is_content: bool) -> None:
            ext.append(token_id)
            ext_indices.append(msg_idx)
            ext_content.append(is_content)

        def emit_text(text: str, msg_idx: int, *, is_content: bool) -> None:
            ids = self._encode(text)
            ext.extend(ids)
            ext_indices.extend([msg_idx] * len(ids))
            ext_content.extend([is_content] * len(ids))

        def emit_segments(segments: list[tuple[str, bool]], msg_idx: int) -> None:
            for tok_id, is_content in attribute_text_segments(
                self._tokenizer, segments
            ):
                ext.append(tok_id)
                ext_indices.append(msg_idx)
                ext_content.append(is_content)

        for i, msg in enumerate(new_messages):
            role = msg.get("role")
            if role == "user":
                emit_special(self._start, i, is_content=False)
                emit_text("user", i, is_content=False)
                emit_special(self._message, i, is_content=False)
                content = _content_text(msg.get("content"))
                if content:
                    emit_segments([(content, True)], i)
                emit_special(self._eot, i, is_content=False)
            elif role == "system":
                emit_special(self._start, i, is_content=False)
                emit_text("system", i, is_content=False)
                emit_special(self._message, i, is_content=False)
                content = _content_text(msg.get("content"))
                if content:
                    emit_segments([(content, True)], i)
                emit_special(self._eot, i, is_content=False)
            elif role == "tool":
                name = self._tool_name(msg)
                emit_special(self._start, i, is_content=False)
                emit_text(f"tool {name}", i, is_content=False)
                emit_special(self._message, i, is_content=False)
                content = _content_text(msg.get("content"))
                emit_segments(
                    [
                        (f'<tool_output name="{name}">\n', False),
                        (content, True),
                        ("\n</tool_output>", False),
                    ],
                    i,
                )
                emit_special(self._eot, i, is_content=False)
            else:
                return None

        # Generation prompt: bare <|start|>assistant
        gen_before = len(ext)
        ext.append(self._start)
        ext.extend(self._encode("assistant"))
        ext_indices.extend([-1] * (len(ext) - gen_before))
        ext_content.extend([False] * (len(ext) - gen_before))

        total_len = len(previous_ids) + len(ext)
        return RenderedTokens(
            token_ids=previous_ids + ext,
            message_indices=[-1] * len(previous_ids) + ext_indices,
            sampled_mask=[False] * total_len,
            is_content=[False] * len(previous_ids) + ext_content,
            message_roles=[m.get("role") or "" for m in new_messages],
            message_tool_names=extract_message_tool_names(new_messages),
        )

    # ── assistant / tool rendering ────────────────────────────────────────
    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        *,
        is_last: bool,
        emit_special,
        emit_text,
        emit_segments,
    ) -> None:
        """Render an assistant turn: reasoning -> tool calls | content.

        Each sub-message is its own harmony frame. The ``<|start|>assistant``
        header (and the ``to=...`` routing) is template scaffolding the
        model continues from at inference — it is NOT model-sampled — so it
        carries ``is_sampled=False`` / ``is_content=False``. The body bytes
        and the terminating ``<|eom|>`` / ``<|eot|>`` are the model's
        sampled emission, so they carry ``is_sampled=True``; on assistant
        turns ``is_content`` mirrors ``sampled_mask`` (the invariant on
        :class:`renderers.RenderedTokens`).
        """
        reasoning = ""
        rc = msg.get("reasoning_content")
        if isinstance(rc, str):
            reasoning = rc

        tool_calls = msg.get("tool_calls") or []
        content = _content_text(msg.get("content"))

        # Count the sub-messages so we know which one is terminal (the last
        # sub-message ends the turn with <|eot|>; earlier ones with <|eom|>).
        n_sub = (1 if reasoning else 0) + len(tool_calls) + (1 if content else 0)
        if n_sub == 0:
            # Empty assistant turn — emit an empty final-content frame so
            # the message contributes at least one attributable token.
            emit_special(self._start, msg_idx, is_sampled=False, is_content=False)
            emit_text("assistant to=user", msg_idx, is_sampled=False, is_content=False)
            emit_special(self._message, msg_idx, is_sampled=False, is_content=False)
            emit_special(self._eot, msg_idx, is_sampled=True, is_content=True)
            return

        sub_seen = 0

        # 1) reasoning (to=self, always closed with <|eom|>)
        if reasoning:
            emit_special(self._start, msg_idx, is_sampled=False, is_content=False)
            emit_text("assistant to=self", msg_idx, is_sampled=False, is_content=False)
            emit_special(self._message, msg_idx, is_sampled=False, is_content=False)
            emit_text(reasoning, msg_idx, is_sampled=True, is_content=True)
            emit_special(self._eom, msg_idx, is_sampled=True, is_content=True)
            sub_seen += 1

        # 2) tool calls (to={fn_name}, ATEM body)
        for tc in tool_calls:
            fn = tc.get("function") or tc
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            sub_seen += 1
            is_terminal = sub_seen == n_sub
            emit_special(self._start, msg_idx, is_sampled=False, is_content=False)
            emit_text(
                f"assistant to={name}", msg_idx, is_sampled=False, is_content=False
            )
            emit_special(self._message, msg_idx, is_sampled=False, is_content=False)
            emit_text(
                atem.render_tool_call(name, args),
                msg_idx,
                is_sampled=True,
                is_content=True,
            )
            emit_special(
                self._eot if is_terminal else self._eom,
                msg_idx,
                is_sampled=True,
                is_content=True,
            )

        # 3) final content (to=user)
        if content:
            sub_seen += 1
            is_terminal = sub_seen == n_sub
            emit_special(self._start, msg_idx, is_sampled=False, is_content=False)
            emit_text(
                "assistant to=user", msg_idx, is_sampled=False, is_content=False
            )
            emit_special(self._message, msg_idx, is_sampled=False, is_content=False)
            emit_text(content, msg_idx, is_sampled=True, is_content=True)
            emit_special(
                self._eot if is_terminal else self._eom,
                msg_idx,
                is_sampled=True,
                is_content=True,
            )

    def _tool_name(self, msg: Message) -> str:
        name = msg.get("name")
        if isinstance(name, str) and name:
            return name
        return "unknown"

    def _render_tool(
        self, msg: Message, msg_idx: int, *, emit_special, emit_segments
    ) -> None:
        """Render a tool-result message.

        Frame: ``<|start|>tool {name}<|message|><tool_output name="{name}">\n
        {content}\n</tool_output><|eot|>``. Tool results are conversation
        history the model never samples, so every token is is_sampled=False;
        only the ``content`` bytes are is_content=True.
        """
        name = self._tool_name(msg)
        content = _content_text(msg.get("content"))
        emit_special(self._start, msg_idx, is_sampled=False, is_content=False)
        # header text ``tool {name}`` — scaffold
        emit_segments([(f"tool {name}", False)], msg_idx, is_sampled=False)
        emit_special(self._message, msg_idx, is_sampled=False, is_content=False)
        emit_segments(
            [
                (f'<tool_output name="{name}">\n', False),
                (content, True),
                ("\n</tool_output>", False),
            ],
            msg_idx,
            is_sampled=False,
        )
        emit_special(self._eot, msg_idx, is_sampled=False, is_content=False)

    # ── parse helpers ─────────────────────────────────────────────────────
    def _strip_stop_tokens(self, ids: list[int]) -> list[int]:
        stop = {self._eot, self._eom, self._eos}
        while ids and ids[-1] in stop:
            ids.pop()
        return ids

    def _iter_frames(self, text: str):
        """Yield ``(header, body)`` for each ``<|start|>...<|message|>...``
        frame in ``text``.

        The header is the text between ``<|start|>`` and ``<|message|>``;
        the body is everything up to the next ``<|start|>`` (stop tokens
        were already stripped). Robust to a leading fragment before the
        first ``<|start|>``.
        """
        START = "<|start|>"
        MSG = "<|message|>"
        # Split into frames on <|start|>.
        chunks = text.split(START)
        for chunk in chunks:
            if not chunk:
                continue
            if MSG not in chunk:
                # A fragment with no <|message|> — could be a bare body
                # (no header). Yield it as an unrouted body.
                yield ("", chunk)
                continue
            header, body = chunk.split(MSG, 1)
            # Trim any trailing <|eom|>/<|eot|>/<|end_of_text|> text that
            # survived decode (skip_special_tokens=False keeps them).
            for tok in ("<|eot|>", "<|eom|>", "<|end_of_text|>"):
                if body.endswith(tok):
                    body = body[: -len(tok)]
            yield (header.strip(), body)

    @staticmethod
    def _recipient_from_header(header: str) -> str | None:
        """Extract the ``to=...`` recipient from a frame header, if any.

        Header examples: ``assistant to=self``, ``assistant to=user``,
        ``assistant to=web_search.query``, ``assistant`` (no recipient).
        """
        if "to=" not in header:
            return None
        after = header.split("to=", 1)[1].strip()
        # Recipient runs to end of header (no spaces in tool names in the
        # reference format beyond the namespace.function dotted form).
        return after.split()[0] if after else None

    def _to_parsed_tool_call(self, atc: "atem.AtemToolCall") -> ParsedToolCall:
        """Map an :class:`atem.AtemToolCall` onto a ``renderers.ParsedToolCall``."""
        status_map = {
            atem.AtemParseStatus.OK: ToolCallParseStatus.OK,
            atem.AtemParseStatus.INVALID_JSON: ToolCallParseStatus.INVALID_JSON,
            atem.AtemParseStatus.MISSING_NAME: ToolCallParseStatus.MISSING_NAME,
            atem.AtemParseStatus.UNCLOSED_BLOCK: ToolCallParseStatus.UNCLOSED_BLOCK,
            atem.AtemParseStatus.MALFORMED_STRUCTURE: (
                ToolCallParseStatus.MALFORMED_STRUCTURE
            ),
        }
        return ParsedToolCall(
            raw=atc.raw,
            name=atc.name,
            arguments=atc.arguments,
            token_span=None,
            status=status_map.get(atc.status, ToolCallParseStatus.MALFORMED_STRUCTURE),
        )


# ── Registration (out-of-tree, no fork of the installed library) ───────────
def register() -> None:
    """Install ``MuseGlimmerRenderer`` into the ``renderers`` registry.

    Mutates the installed library's registry dicts at import time so that
    ``renderers.create_renderer(...)`` / ``AutoRendererConfig`` resolve the
    model name(s) to this renderer WITHOUT forking the library:

      * ``renderers.base.RENDERER_REGISTRY[name] = MuseGlimmerRenderer`` for
        each of ``"muse_glimmer"`` / ``"glimmer"``.
      * ``renderers.configs._CONFIG_BY_NAME[name] = MuseGlimmerRendererConfig``
        for the same names, so ``_config_class_for`` / ``config_from_name``
        resolve them.
      * ``renderers.base.MODEL_RENDERER_MAP[model] = "muse_glimmer"`` for the
        known checkpoint name(s), so ``AutoRendererConfig`` picks it up from
        ``tokenizer.name_or_path``.

    Idempotent — safe to call multiple times.
    """
    import renderers.base as _base
    import renderers.configs as _configs

    # Ensure the built-in registry is populated first so we only add.
    _base._populate_registry()

    for name in (_RENDERER_NAME, *_ALIAS_NAMES):
        _base.RENDERER_REGISTRY[name] = MuseGlimmerRenderer
        _configs._CONFIG_BY_NAME[name] = MuseGlimmerRendererConfig

    for model in _MODEL_NAMES:
        _base.MODEL_RENDERER_MAP[model] = _RENDERER_NAME


__all__ = [
    "MuseGlimmerRenderer",
    "MuseGlimmerRendererConfig",
    "register",
]
