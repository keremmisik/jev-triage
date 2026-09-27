"""Confidence-gated routing: turns calibrated probabilities into an action.

This is the "code stays in control" half of the System One pattern. The model
answers small questions; this module owns the policy, and every threshold is a
named, testable number instead of a sentence in a prompt.

Decision order (first match wins):

1. **PAGE** if the probability mass on {critical, high} is at least
   ``page_risk_threshold`` *and* there is user impact or a required action.
   This check uses probability mass rather than the arg-max label, so an event
   the model thinks is 45% critical + 20% high still pages. Missing an outage
   costs more than a false page, so this rule runs *before* the confidence gate.
2. **HUMAN_REVIEW** if severity confidence is below ``review_min_confidence``
   (the model is saying "I don't know"). Low *subsystem* confidence only
   matters when someone has to act, because the subsystem decides who owns it;
   an informational event with an unclear subsystem is not a human's problem.
3. **SUPPRESS** confident noise, in two shapes: (a) low/info *transient*
   conditions that heal themselves, and (b) *routine* info messages (cron runs,
   session starts) that are clearly not actionable. Degraded (fallback) and
   security-relevant results are never suppressed.
4. **TICKET** if an action is needed.
5. **NOTIFY** otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .models import Classification, Route


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    review_min_confidence: float = 0.50
    auto_min_confidence: float = 0.80
    page_risk_threshold: float = 0.60
    impact_threshold: float = 0.50
    actionable_threshold: float = 0.50
    suppress_transient_threshold: float = 0.75
    routine_max_actionable: float = 0.20

    @classmethod
    def from_settings(cls, s: Settings) -> RoutingPolicy:
        return cls(
            review_min_confidence=s.review_min_confidence,
            auto_min_confidence=s.auto_min_confidence,
            page_risk_threshold=s.page_risk_threshold,
            impact_threshold=s.impact_threshold,
            actionable_threshold=s.actionable_threshold,
            suppress_transient_threshold=s.suppress_transient_threshold,
            routine_max_actionable=s.routine_max_actionable,
        )


def decide(c: Classification, policy: RoutingPolicy) -> tuple[Route, list[str]]:
    """Return the route and human-readable reasons (for audit logs and the UI)."""
    sev = c.severity
    risk = sev.p("critical") + sev.p("high")
    impact = c.nouls.get("user_impact", 0.0)
    actionable = c.nouls.get("actionable", 0.0)
    transient = c.nouls.get("transient", 0.0)
    security = c.nouls.get("security_relevant", 0.0)

    reasons: list[str] = []
    if c.degraded:
        reasons.append("degraded: fallback classifier used")

    # 1. Page on high risk mass + impact/action, regardless of confidence.
    if risk >= policy.page_risk_threshold and (
        impact >= policy.impact_threshold or actionable >= policy.actionable_threshold
    ):
        reasons.append(f"P(critical|high)={risk:.2f} >= {policy.page_risk_threshold}")
        reasons.append(f"user_impact={impact:.2f}, actionable={actionable:.2f}")
        if sev.confidence < policy.review_min_confidence:
            reasons.append("low severity confidence; paging anyway (safety bias)")
        return Route.PAGE, reasons

    # 2. Model is unsure: a human decides.
    needs_owner = actionable >= policy.actionable_threshold
    low: list[str] = []
    if sev.confidence < policy.review_min_confidence:
        low.append(f"severity confidence {sev.confidence:.2f} < {policy.review_min_confidence}")
    if needs_owner and c.subsystem.confidence < policy.review_min_confidence:
        low.append(
            f"subsystem confidence {c.subsystem.confidence:.2f} < "
            f"{policy.review_min_confidence} for an actionable event"
        )
    if low:
        return Route.HUMAN_REVIEW, reasons + low

    # Security-relevant events are never suppressed and always get a ticket.
    if security >= policy.actionable_threshold:
        reasons.append(f"security_relevant={security:.2f}")
        return Route.TICKET, reasons

    # 3. Suppress only confident noise (security-relevant events returned above).
    confident_benign = (
        not c.degraded
        and sev.confidence >= policy.auto_min_confidence
        and actionable < policy.actionable_threshold
        and impact < policy.impact_threshold
    )
    # 3a. Self-healing conditions (a retry, a brief spike).
    if (
        confident_benign
        and sev.label in {"low", "info"}
        and transient >= policy.suppress_transient_threshold
    ):
        reasons.append(
            f"benign ({sev.label}, conf {sev.confidence:.2f}), transient={transient:.2f}"
        )
        return Route.SUPPRESS, reasons
    # 3b. Routine operation (cron ran, session started): nothing is wrong at all.
    if confident_benign and sev.label == "info" and actionable < policy.routine_max_actionable:
        reasons.append(f"routine (info, conf {sev.confidence:.2f}, actionable={actionable:.2f})")
        return Route.SUPPRESS, reasons

    # 4 / 5.
    if actionable >= policy.actionable_threshold:
        reasons.append(f"actionable={actionable:.2f}")
        return Route.TICKET, reasons
    reasons.append(f"no action needed (actionable={actionable:.2f})")
    return Route.NOTIFY, reasons
