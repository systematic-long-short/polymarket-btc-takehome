"""The single reference baseline — every candidate is scored against this.

Implementation lives in ``polybench.baselines``; this file re-exports it so
candidates can read a short file to see exactly what they're benchmarked
against.

    MomentumBaseline — 30s BTC momentum, trades dynamically — THE BAR.
"""

from polybench.baselines import MomentumBaseline

__all__ = ["MomentumBaseline"]
