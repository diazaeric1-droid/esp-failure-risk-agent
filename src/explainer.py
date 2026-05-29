"""Turn model feature contributions into plain-English explanations via Claude."""
from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv


EXPLAINER_SYSTEM_PROMPT = """You are a Senior Production Engineer writing the rationale section of an ESP failure-risk digest. Given a well's current diagnostics and the top features driving its risk score, write 2-3 sentences (max 60 words) explaining WHY this well is high-risk in the language a field engineer uses.

Style rules:
- Lead with the failure mode you suspect (e.g., "downthrust from pump-off", "gas-lock signature", "scale buildup")
- Reference specific values that triggered the call (e.g., "intake pressure at 28 psi, declining")
- End with a concrete next step (chemical treatment, ESP pull, VSD adjustment, gas separator)
- No filler, no hedging, no "based on the model"
"""


def explain_well(
    well_id: str,
    risk_score: float,
    feature_values: dict[str, float],
    top_drivers: list[tuple[str, float]],
    model: str = "claude-sonnet-4-6",
    client: Anthropic | None = None,
) -> str:
    """Generate a plain-English rationale for a single high-risk well."""
    if client is None:
        load_dotenv()
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    drivers_str = "\n".join(
        f"  - {feat}: contribution={contrib:+.2f}, current_value={feature_values.get(feat, 'n/a')}"
        for feat, contrib in top_drivers
    )

    prompt = f"""Well: {well_id}
30-day failure probability: {risk_score:.1%}

Top features driving this risk (SHAP-style contributions, positive = increases risk):
{drivers_str}

Write the 2-3 sentence rationale."""

    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=EXPLAINER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def top_drivers(contribs_row, k: int = 4) -> list[tuple[str, float]]:
    """Pick the top-k features by absolute contribution (excluding bias)."""
    s = contribs_row.drop("bias")
    return list(s.abs().sort_values(ascending=False).head(k).items())
