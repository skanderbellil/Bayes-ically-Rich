"""
Claude-backed council specialists (optional backend).

Each (period, domain) assessment is one Messages API call constrained to a
JSON schema: {stance, confidence, rationale}. The specialist sees ONLY the
blinded DomainView — no dates, tickers, or levels — and the system prompt
explicitly forbids guessing the historical episode.

For large period counts, `run_llm_council_batch()` submits every
(period, domain, original|mirrored) assessment as one Message Batch
(50% cost, all requests independent — exactly the council's shape).

Requires ANTHROPIC_API_KEY (loaded from .env via posterioralpha.env) and
`pip install anthropic`.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Tuple

from posterioralpha.council.blinding import PeriodContext
from posterioralpha.council.specialists import Verdict
from posterioralpha.env import load_env

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"

_SCHEMA = {
    "type": "object",
    "properties": {
        "stance": {
            "type": "number",
            "description": "-1 = maximally risk-off, +1 = maximally risk-on",
        },
        "confidence": {
            "type": "number",
            "description": "0 = no signal, 1 = very clear signal",
        },
        "rationale": {"type": "string"},
    },
    "required": ["stance", "confidence", "rationale"],
    "additionalProperties": False,
}

_SYSTEM = """You are a {domain} analyst on an investment committee. You are
shown an anonymized slice of market data: rolling z-scores of {domain}
indicators and the underlying asset's normalized return path. There are no
dates, no instrument names, and no absolute levels — deliberately. You must
NOT try to identify which historical episode this is, and you must not use
any recollection of historical market events. Judge ONLY the statistical
configuration in front of you, as a careful analyst would judge a market
state they have never seen labeled.

Assess whether this configuration argues for being invested in the risky
asset over the next month. Respond with stance in [-1, 1] (-1 = fully
risk-off, +1 = fully risk-on), confidence in [0, 1], and a one-sentence
rationale grounded in the supplied numbers only."""


def _user_message(ctx: PeriodContext, domain: str) -> str:
    view = ctx.views[domain]
    lines = [f"Period #{ctx.period_id} (relative time; oldest → newest values)."]
    for label, vals in view.indicators.items():
        lines.append(f"{label}: {vals}")
    lines.append(f"asset normalized cumulative path: {ctx.asset_path}")
    lines.append(f"asset realized-vol z-score: {ctx.asset_vol_z}")
    return "\n".join(lines)


def _parse_verdict(text: str) -> Verdict:
    data = json.loads(text)
    return Verdict(
        stance=float(max(-1.0, min(1.0, data["stance"]))),
        confidence=float(max(0.0, min(1.0, data["confidence"]))),
        rationale=str(data.get("rationale", ""))[:300],
    )


def _client():
    load_env()
    import os
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "LLM council needs an API key: add ANTHROPIC_API_KEY=... to <repo>/.env"
        )
    import anthropic
    return anthropic.Anthropic()


class LLMSpecialist:
    """Synchronous Claude-backed specialist (fine for small period counts)."""

    def __init__(self, domain: str, model: str = DEFAULT_MODEL):
        self.domain = domain
        self.model = model
        self._client = _client()

    def assess(self, ctx: PeriodContext) -> Verdict:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=_SYSTEM.format(domain=self.domain),
            messages=[{"role": "user", "content": _user_message(ctx, self.domain)}],
            output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return _parse_verdict(text)


def run_llm_council_batch(
    contexts: List[PeriodContext],
    domains: List[str],
    model: str = DEFAULT_MODEL,
    include_mirrors: bool = True,
    poll_seconds: int = 30,
) -> Dict[Tuple[int, str, bool], Verdict]:
    """
    Assess every (period, domain[, mirrored]) in one Message Batch.

    Returns {(period_id, domain, is_mirrored): Verdict}. Batches cost 50% of
    standard pricing and every council assessment is independent, so this is
    the natural execution mode for a full backtest.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = _client()

    requests = []
    for ctx in contexts:
        variants = [ctx, ctx.mirror()] if include_mirrors else [ctx]
        for v in variants:
            for domain in domains:
                if domain not in v.views:
                    continue
                requests.append(Request(
                    custom_id=f"p{v.period_id}|{domain}|{'m' if v.is_mirrored else 'o'}",
                    params=MessageCreateParamsNonStreaming(
                        model=model,
                        max_tokens=1024,
                        system=_SYSTEM.format(domain=domain),
                        messages=[{"role": "user",
                                   "content": _user_message(v, domain)}],
                        output_config={"format": {"type": "json_schema",
                                                  "schema": _SCHEMA}},
                    ),
                ))

    logger.info("submitting council batch: %d assessments (%s)", len(requests), model)
    batch = client.messages.batches.create(requests=requests)

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        logger.info("batch %s: %s (%d processing)", batch.id,
                    batch.processing_status, batch.request_counts.processing)
        time.sleep(poll_seconds)

    verdicts: Dict[Tuple[int, str, bool], Verdict] = {}
    for result in client.messages.batches.results(batch.id):
        pid_s, domain, flag = result.custom_id.split("|")
        key = (int(pid_s[1:]), domain, flag == "m")
        if result.result.type == "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "{}")
            try:
                verdicts[key] = _parse_verdict(text)
            except Exception:
                verdicts[key] = Verdict(0.0, 0.0, "unparseable")
        else:
            verdicts[key] = Verdict(0.0, 0.0, f"batch:{result.result.type}")
    return verdicts
