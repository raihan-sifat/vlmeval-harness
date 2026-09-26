"""Metrics.

Each function takes the item-level outcomes for a run and returns
``{name: MetricSummary}``. Keeping the mapping from items to numbers in one
place means every task and every report agrees on what "accuracy" means, and
that the same confidence-interval machinery wraps all of them.
"""

from .binary import yes_no_metrics
from .mcq import accuracy_metric, choice_distribution, consistency_metrics

__all__ = [
    "accuracy_metric",
    "yes_no_metrics",
    "choice_distribution",
    "consistency_metrics",
]
