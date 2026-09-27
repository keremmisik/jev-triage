from __future__ import annotations

import pytest

from jev_triage.classifier import normalized_confidence
from jev_triage.models import SEVERITIES, SUBSYSTEMS, ChoiceResult, Classification, Route
from jev_triage.router import RoutingPolicy, decide

POLICY = RoutingPolicy()


def choice(probs: dict[str, float], space: tuple[str, ...]) -> ChoiceResult:
    full = {k: 0.0 for k in space} | probs
    return ChoiceResult(
        label=max(full, key=full.__getitem__),
        confidence=normalized_confidence(full),
        probabilities=full,
    )


def cls(
    sev: dict[str, float],
    sub: dict[str, float] | None = None,
    degraded: bool = False,
    **nouls: float,
) -> Classification:
    n = {"user_impact": 0.0, "actionable": 0.0, "transient": 0.0, "security_relevant": 0.0}
    n.update(nouls)
    return Classification(
        severity=choice(sev, SEVERITIES),
        subsystem=choice(sub or {"storage": 1.0}, SUBSYSTEMS),
        nouls=n,
        classifier="test",
        degraded=degraded,
    )


def test_confident_critical_with_impact_pages() -> None:
    route, reasons = decide(cls({"critical": 1.0}, user_impact=0.9), POLICY)
    assert route is Route.PAGE and any("P(critical|high)" in r for r in reasons)


def test_split_high_risk_mass_still_pages_even_if_unsure() -> None:
    # arg-max is "critical" but only at 0.35; risk mass is 0.35+0.30=0.65.
    c = cls({"critical": 0.35, "high": 0.30, "medium": 0.2, "low": 0.15}, actionable=0.8)
    assert c.severity.confidence < POLICY.review_min_confidence
    route, reasons = decide(c, POLICY)
    assert route is Route.PAGE
    assert any("safety bias" in r for r in reasons)


def test_high_risk_without_impact_or_action_is_not_paged() -> None:
    route, _ = decide(cls({"high": 1.0}, user_impact=0.1, actionable=0.1), POLICY)
    assert route is Route.NOTIFY


def test_uncertain_goes_to_human() -> None:
    c = cls({"medium": 0.4, "low": 0.35, "info": 0.25}, actionable=0.6)
    route, reasons = decide(c, POLICY)
    assert route is Route.HUMAN_REVIEW and "severity confidence" in reasons[-1]


def test_uncertain_subsystem_goes_to_human() -> None:
    c = cls({"medium": 1.0}, sub={"network": 0.3, "dns": 0.3, "web_app": 0.4}, actionable=0.9)
    assert decide(c, POLICY)[0] is Route.HUMAN_REVIEW


def test_security_never_suppressed() -> None:
    c = cls({"info": 1.0}, transient=0.95, security_relevant=0.8)
    assert decide(c, POLICY)[0] is Route.TICKET


def test_benign_transient_suppressed() -> None:
    assert decide(cls({"info": 1.0}, transient=0.9), POLICY)[0] is Route.SUPPRESS


def test_degraded_is_never_suppressed() -> None:
    route, reasons = decide(cls({"info": 1.0}, transient=0.9, degraded=True), POLICY)
    assert route is Route.NOTIFY and reasons[0].startswith("degraded")


@pytest.mark.parametrize(("actionable", "expected"), [(0.8, Route.TICKET), (0.2, Route.NOTIFY)])
def test_ticket_vs_notify(actionable: float, expected: Route) -> None:
    assert decide(cls({"medium": 1.0}, actionable=actionable), POLICY)[0] is expected


def test_normalized_confidence_bounds() -> None:
    assert normalized_confidence({"a": 1.0, "b": 0.0}) == 1.0
    assert normalized_confidence({"a": 0.5, "b": 0.5}) == 0.0
    assert normalized_confidence({"a": 1.0}) == 1.0


def test_routine_info_is_suppressed() -> None:
    # Cron ran / session started: confident info, clearly not actionable.
    route, reasons = decide(cls({"info": 1.0}, actionable=0.05), POLICY)
    assert route is Route.SUPPRESS and reasons[-1].startswith("routine")


def test_unclear_subsystem_only_matters_when_actionable() -> None:
    unsure_sub = {"auth_access": 0.4, "scheduler_jobs": 0.35, "other": 0.25}
    # Informational: nobody needs to own it, so no human review.
    assert decide(cls({"info": 1.0}, sub=unsure_sub, actionable=0.05), POLICY)[0] is Route.SUPPRESS
    # Actionable: routing needs an owner, so a human decides.
    route, reasons = decide(cls({"medium": 1.0}, sub=unsure_sub, actionable=0.8), POLICY)
    assert route is Route.HUMAN_REVIEW and "actionable event" in reasons[-1]


def test_routine_suppression_respects_impact_and_actionability() -> None:
    assert decide(cls({"info": 1.0}, actionable=0.3), POLICY)[0] is Route.NOTIFY
    assert decide(cls({"info": 1.0}, actionable=0.05, user_impact=0.7), POLICY)[0] is Route.NOTIFY
