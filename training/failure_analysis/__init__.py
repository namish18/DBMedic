"""Failure collection, classification, and verified correction engine (Phase 4)."""

from training.failure_analysis.analyzer import TrajectoryFailureAnalyzer
from training.failure_analysis.classifier import (
    FAILURE_CATEGORIES,
    FailureClassifier,
    FailureRecord,
)
from training.failure_analysis.correction_generator import (
    CorrectionGenerator,
    CorrectionVerificationResult,
)

__all__ = [
    "FAILURE_CATEGORIES",
    "FailureRecord",
    "FailureClassifier",
    "TrajectoryFailureAnalyzer",
    "CorrectionGenerator",
    "CorrectionVerificationResult",
]

