# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A stand-in retrieval server for the Muse Glimmer Search-R1 pipeline POC.

Purpose (and honest caveats)
----------------------------
The real Search-R1 recipe expects a dense retrieval server (e5 index over
wiki-18) listening on ``/retrieve``. Standing that up needs the FAISS index +
corpus (tens of GB) and spare GPUs. To let anyone smoke-test the *end-to-end RL
plumbing* (renderer -> generator -> tool call -> env -> reward -> trainer)
without that infrastructure, this server implements the same HTTP contract but
returns synthetic passages.

It has **two modes**:

* ``--mode oracle`` (default, POC): each returned passage is constructed to
  contain a plausible answer span for the query. When paired with the
  Search-R1 dataset (whose rows carry ``golden_answers``) via
  ``--answers-parquet``, it returns the *golden* answer verbatim inside a
  passage. This makes reward achievable so the pipeline produces gradient — it
  proves the loop, **not** convergence or real retrieval quality. This mirrors
  the documented POC shortcut and MUST NOT be used to claim model quality.
* ``--mode empty``: returns empty passages (useful to measure the closed-book
  baseline / confirm the tool wiring without leaking answers).

HTTP contract (matches ``search_r1/env.py``)
--------------------------------------------
``POST /retrieve`` with body ``{"queries": [str, ...], "topk": int,
"return_scores": bool}`` -> ``{"result": [[{"contents": str}, ...], ...]}``
(one inner list per query, each a list of ``topk`` passage dicts).

Usage
-----
    python -m torchtitan.experiments.rl.examples.glimmer_search_r1.standin_retriever \
        --host 127.0.0.1 --port 8000 --mode oracle \
        --answers-parquet <dir-or-file with question/golden_answers>

If ``--answers-parquet`` is omitted, oracle mode falls back to echoing the query
with a generic templated passage (still exercises the loop, but reward will be
near zero because no golden span is present).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger("glimmer_standin_retriever")

# question (normalized) -> one golden answer string
_ANSWER_BY_QUESTION: dict[str, str] = {}

# Stopwords stripped before fuzzy query<->question matching so a *reworded*
# search query ("author of the novel Frankenstein") still resolves to the
# dataset question ("Who wrote the novel Frankenstein?") and surfaces its gold.
_STOPWORDS = frozenset(
    "a an the of to in on for is are was were who what when where which why how "
    "did do does done which whom whose that this these those and or by with as "
    "at from into it its name novel play call called write wrote written".split()
)


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _content_words(q: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", _normalize(q))
    return {t for t in toks if t not in _STOPWORDS and len(t) > 1}


def _best_gold_for_query(query: str) -> str | None:
    """Exact-normalized lookup, else best content-word-overlap match.

    Returns the golden answer of the known question that shares the most
    content words with ``query`` (Jaccard), provided the overlap clears a
    minimum bar. Makes the oracle robust to the model rewording the question
    into a search query, so a genuine search reliably surfaces gold (the
    signal ``RewardExactMatch(retrieval_score=...)`` grades on).
    """
    exact = _ANSWER_BY_QUESTION.get(_normalize(query))
    if exact is not None:
        return exact
    qw = _content_words(query)
    if not qw:
        return None
    best_gold: str | None = None
    best_score = 0.0
    for known_q, gold in _QUESTION_WORDS.items():
        kw = gold[1]
        if not kw:
            continue
        inter = len(qw & kw)
        if inter == 0:
            continue
        jacc = inter / len(qw | kw)
        # Also require covering a good fraction of the (shorter) question.
        cover = inter / min(len(qw), len(kw))
        score = 0.5 * jacc + 0.5 * cover
        if score > best_score:
            best_score = score
            best_gold = gold[0]
    # Threshold chosen so a reworded query matches but unrelated queries don't.
    return best_gold if best_score >= 0.5 else None


# normalized question -> (gold_answer, content_word_set), for fuzzy matching
_QUESTION_WORDS: dict[str, tuple[str, set[str]]] = {}



def _load_answers(parquet_path: str) -> int:
    """Populate the question->golden-answer map from a Search-R1 parquet.

    Accepts a local parquet file or an HF ``repo_id`` + filename handled by
    ``datasets``. Best-effort: any load failure leaves the map empty (oracle
    mode then degrades to generic passages).
    """
    try:
        from datasets import load_dataset

        if parquet_path.endswith(".parquet"):
            ds = load_dataset("parquet", data_files=parquet_path, split="train")
        else:
            # treat as a directory containing train.parquet
            ds = load_dataset(
                "parquet", data_files=f"{parquet_path}/train.parquet", split="train"
            )
        n = 0
        for row in ds:
            q = _normalize(str(row["question"]))
            gold = row["golden_answers"]
            if hasattr(gold, "tolist"):  # numpy array from parquet
                gold = gold.tolist()
            if isinstance(gold, (list, tuple)) and len(gold):
                gold_str = str(gold[0])
                _ANSWER_BY_QUESTION[q] = gold_str
                _QUESTION_WORDS[q] = (gold_str, _content_words(str(row["question"])))
                n += 1
        logger.info("loaded %d question->answer pairs for oracle mode", n)
        return n
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load answers parquet %s: %s", parquet_path, exc)
        return 0


def _oracle_passage(query: str, rank: int) -> dict[str, str]:
    gold = _best_gold_for_query(query)
    if gold is not None:
        contents = (
            f'"{query.strip()}" \u2014 According to reference sources, '
            f"the answer is {gold}. (stand-in passage #{rank + 1})"
        )
    else:
        contents = (
            f"Reference passage #{rank + 1} related to: {query.strip()}. "
            "(stand-in oracle had no golden answer for this query)"
        )
    return {"contents": contents}


def _empty_passage(query: str, rank: int) -> dict[str, str]:
    return {"contents": ""}


class _Handler(BaseHTTPRequestHandler):
    # set by main()
    mode = "oracle"

    def log_message(self, *args: Any) -> None:  # silence default noisy logging
        pass

    def do_POST(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path.rstrip("/") not in ("/retrieve", ""):
            self.send_error(404, "only POST /retrieve is supported")
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON body")
            return

        queries = body.get("queries") or []
        topk = int(body.get("topk", 3))
        make = _oracle_passage if self.mode == "oracle" else _empty_passage
        result = [[make(q, r) for r in range(topk)] for q in queries]

        payload = json.dumps({"result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Stand-in retrieval server for the Glimmer Search-R1 POC.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", choices=["oracle", "empty"], default="oracle")
    ap.add_argument(
        "--answers-parquet",
        default=None,
        help="local parquet (or dir with train.parquet) with question/golden_answers; "
        "enables true-answer oracle passages in --mode oracle.",
    )
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_arg_parser().parse_args()
    if args.mode == "oracle" and args.answers_parquet:
        _load_answers(args.answers_parquet)
    _Handler.mode = args.mode
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    logger.info(
        "stand-in retriever (mode=%s) listening on http://%s:%d/retrieve",
        args.mode,
        args.host,
        args.port,
    )
    if args.mode == "oracle":
        logger.warning(
            "ORACLE MODE returns golden answers inside passages: proves the RL "
            "pipeline, NOT retrieval quality or model convergence."
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
