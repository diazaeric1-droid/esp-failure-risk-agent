"""Turn model feature contributions into plain-English explanations via Claude.

Design principle (shared with the Daily Production Digest): **detection stays
deterministic, the LLM only narrates.** A rule-based classifier maps the engineered
features to a suspected ESP failure mode FIRST; the LLM is then asked to write the
rationale *for that mode*, so it can't hallucinate a diagnosis the data doesn't
support.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv


class MissingAPIKey(RuntimeError):
    """Raised when an LLM call is attempted without ANTHROPIC_API_KEY set."""


EXPLAINER_SYSTEM_PROMPT = """You are a Senior Production Engineer writing the rationale section of an ESP failure-risk digest. You are given a well's diagnostics, the top features driving its risk score, and a SUSPECTED FAILURE MODE already determined by a deterministic classifier. Write 2-3 sentences (max 60 words) explaining WHY this well is high-risk in the language a field engineer uses.

Style rules:
- Lead with the suspected failure mode you were given (do NOT contradict it; you may add nuance).
- Reference the specific values that triggered the call (e.g., "intake pressure at 28 psi, declining").
- End with a concrete next step (chemical/scale treatment, ESP pull, VSD/frequency adjustment, gas separator, motor/cable megger test).
- No filler, no hedging, no "based on the model".
"""


# ---- deterministic failure-mode classifier --------------------------------

def classify_failure_mode(feature_values: dict[str, float]) -> tuple[str, str]:
    """Map an engineered feature row to a suspected ESP failure mode + evidence.

    Returns (mode_label, evidence). Rules are ordered by specificity; the first
    match wins. This is intentionally simple and auditable — it grounds the LLM
    narration and is shown to the user alongside the risk score.
    """
    f = feature_values
    g = lambda k, d=0.0: float(f.get(k, d))

    imb_max = g("current_imbalance_max_30d", 3.0)
    imb_days = g("high_imbalance_days_30d")
    bfpd_cv = g("bfpd_cv_30d")
    downtime = g("downtime_days_30d")
    low_intake_days = g("low_intake_days_30d")
    intake_slope = g("intake_p_slope_30d")
    intake_mean = g("intake_p_last7_mean", 130.0)
    amps_slope = g("motor_amps_slope_30d")
    temp_slope = g("motor_temp_slope_30d")
    bfpd_slope = g("bfpd_slope_30d")
    runtime_mean = g("runtime_last7_mean", 99.0)
    freq_slope = g("drive_freq_slope_30d")

    # 1) Electrical / current imbalance — the imbalance channel is the tell.
    if imb_max >= 9 or imb_days >= 2:
        return ("Electrical — current imbalance / incipient motor short",
                f"current imbalance peaked at {imb_max:.0f}% ({int(imb_days)} day(s) >8%); "
                f"megger the motor and cable before the next start.")

    # 2) Gas lock / pump-off cycling — flow volatility + runtime cycling, freq rising.
    if bfpd_cv >= 0.15 and downtime >= 3:
        return ("Gas lock — pump-off cycling",
                f"production volatile (CV {bfpd_cv:.0%}) with {int(downtime)} low-runtime day(s)"
                + (f" and drive frequency climbing {freq_slope:+.2f} Hz/d" if freq_slope > 0.05 else "")
                + "; review gas handling / install a separator and check VSD pump-off setpoints.")

    # 3) Gas interference — smooth intake collapse.
    if low_intake_days >= 2 or (intake_slope <= -1.0 and intake_mean < 90):
        return ("Gas interference — intake pressure collapse",
                f"intake at {intake_mean:.0f} psi, trending {intake_slope:+.1f} psi/d; "
                f"reduce drawdown / adjust frequency and evaluate a gas separator.")

    # 4) Scale / abrasive buildup — amps & temp creeping together, intake holding.
    if amps_slope >= 0.15 and temp_slope >= 0.10:
        return ("Scale / abrasive buildup",
                f"motor amps creeping {amps_slope:+.2f} A/d and temperature {temp_slope:+.2f} °F/d "
                f"with stable intake; schedule a scale-inhibitor squeeze / acid treatment.")

    # 5) Downthrust / reservoir decline — rate slumping, runtime down, amps not rising.
    if bfpd_slope <= -8 and runtime_mean < 96 and amps_slope < 0.15:
        return ("Downthrust / declining inflow",
                f"production sliding {bfpd_slope:+.0f} bbl/d with runtime at {runtime_mean:.0f}%; "
                f"verify pump is within POR and consider a re-rate or smaller stage count.")

    return ("Unclear — multiple weak signals",
            "no single dominant signature; review the trend plot and recent well work before acting.")


def explain_well(
    well_id: str,
    risk_score: float,
    feature_values: dict[str, float],
    top_drivers: list[tuple[str, float]],
    suspected_mode: str | None = None,
    model: str = "claude-sonnet-4-6",
    client=None,
) -> str:
    """Generate a plain-English rationale for a single high-risk well.

    Raises ``MissingAPIKey`` (not a bare KeyError) when no client is supplied and
    ANTHROPIC_API_KEY is unset, so callers can degrade gracefully.
    """
    if suspected_mode is None:
        suspected_mode = classify_failure_mode(feature_values)[0]

    if client is None:
        load_dotenv()
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKey(
                "ANTHROPIC_API_KEY is not set — set it to generate AI rationales. "
                "The risk scores, drivers, and suspected failure mode work without it.")
        from anthropic import Anthropic
        client = Anthropic(api_key=key)

    drivers_str = "\n".join(
        f"  - {feat}: contribution={contrib:+.2f}, current_value={feature_values.get(feat, 'n/a')}"
        for feat, contrib in top_drivers
    )

    prompt = f"""Well: {well_id}
30-day failure probability: {risk_score:.1%}
Suspected failure mode (deterministic classifier): {suspected_mode}

Top features driving this risk (XGBoost Tree SHAP contributions, positive = increases risk):
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
