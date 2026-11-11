"""
agent/audit/__init__.py
========================
Phase 5: AuditKIT integration package for DBMedic.

Exports the primary adapter, builders, and evaluators.
AuditKIT (Lexsi Labs, LSAL-1.2) is used as the evaluation spine.
If auditkit is not installed, a lightweight stub is used automatically.
"""

from agent.audit.auditkit_adapter import AuditKITAdapter, AuditKITStub
from agent.audit.episode_builder import EpisodeBuilder
from agent.audit.outcome_verifier import OutcomeVerifier

__all__ = [
    "AuditKITAdapter",
    "AuditKITStub",
    "EpisodeBuilder",
    "OutcomeVerifier",
]

