"""The Jev question set.

Design rules, taken from TypeSafe's guidance for jev-1.13:

* **Atomic questions.** Each question asks one gut-check judgment. Combining
  them (e.g. "is this worth paging for?") happens in ``router.py``, in code,
  where the weights are explicit and testable.
* **Literal wording.** Jev reads instructions literally, so every option has
  an explicit description with boundary cases.
* **No arithmetic in the model.** Counting, rates and time windows are
  computed in code before the call (see ``preprocess.py``).
* **One call per event.** All questions go in one request; Jev evaluates them
  in parallel, so extra questions barely change latency.

Question keys are part of the public contract: the research harness and the
router both reference them, so rename with care.
"""

from __future__ import annotations

from typing import Final

from typesafe_sdk import Choice, Noul, Question

from .models import SEVERITIES, SUBSYSTEMS

# Short, operator-oriented rubric for each severity level.
SEVERITY_CRITERIA: Final[dict[str, str]] = {
    "critical": (
        "A production service is down, data loss is occurring or imminent, "
        "or a security breach is in progress. Needs action within minutes."
    ),
    "high": (
        "Serious degradation or a failure that will become an outage soon if "
        "ignored (e.g. disk almost full, repeated crashes, replication broken)."
    ),
    "medium": (
        "A real problem with limited impact or a workaround; should be fixed within a working day."
    ),
    "low": "Minor issue or warning with no current user impact.",
    "info": "Normal operation, routine status message, or debug output. Nothing is wrong.",
}

SUBSYSTEM_CRITERIA: Final[dict[str, str]] = {
    "storage": "Disks, filesystems, mounts, RAID, I/O errors, inode or space exhaustion.",
    "memory": "RAM, swap, OOM killer, memory leaks, allocation failures.",
    "cpu_load": "CPU saturation, load average, throttling, runaway processes.",
    "network": "Interfaces, packet loss, TCP connection errors, timeouts, TLS handshakes.",
    "dns": "Name resolution failures, DNS server errors, NXDOMAIN, resolver timeouts.",
    "auth_access": "Logins, SSH, sudo, PAM, LDAP/AD, permissions, tokens, certificates for auth.",
    "database": "Database servers, queries, replication, connection pools, locks, deadlocks.",
    "web_app": "Web servers, reverse proxies, HTTP 4xx/5xx, application exceptions.",
    "kernel_hardware": "Kernel messages, drivers, hardware faults, ECC, temperature, firmware.",
    "scheduler_jobs": "Cron, systemd timers, batch jobs, backups, CI jobs.",
    "other": "None of the categories above clearly applies.",
}

# Sanity check at import time: rubric keys must match the label spaces exactly.
assert tuple(SEVERITY_CRITERIA) == SEVERITIES, "severity rubric out of sync with SEVERITIES"
assert tuple(SUBSYSTEM_CRITERIA) == SUBSYSTEMS, "subsystem rubric out of sync with SUBSYSTEMS"

# Yes/no signals. Keys are referenced by router.py and the research harness.
NOUL_QUESTIONS: Final[dict[str, Noul]] = {
    "user_impact": Noul(
        instructions=(
            "Does the `event` indicate that end users or dependent services are affected right now?"
        ),
        criteria={
            "true": "Users or other services are currently failing or degraded.",
            "false": "No current impact on users or other services is indicated.",
        },
    ),
    "actionable": Noul(
        instructions="Does the `event` require a human operator to take an action?",
        criteria={
            "true": "Someone must investigate, fix, or change something.",
            "false": "It is informational or will be handled automatically.",
        },
    ),
    "transient": Noul(
        instructions=(
            "Is the `event` most likely a transient condition that resolves on its own, "
            "such as a single retry, a brief spike, or an expected restart?"
        ),
    ),
    "security_relevant": Noul(
        instructions=(
            "Is the `event` relevant to security, such as failed or suspicious "
            "authentication, privilege changes, or signs of an attack?"
        ),
    ),
}


def build_questions() -> dict[str, Question]:
    """Return the full question map sent with every triage request.

    A fresh dict is returned each time so callers may add experiment-specific
    questions without mutating module state.
    """
    questions: dict[str, Question] = {
        "severity": Choice(
            instructions="How severe is the `event` for the operations team?",
            criteria=SEVERITY_CRITERIA,
        ),
        "subsystem": Choice(
            instructions="Which subsystem is the `event` primarily about?",
            criteria=SUBSYSTEM_CRITERIA,
        ),
    }
    questions.update(NOUL_QUESTIONS)
    return questions
