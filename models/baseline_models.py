"""Reference baselines — the three benchmarks a candidate is compared against.

Implementations live in ``polybench.baselines``; this file re-exports them
so candidates can read a single short file to see what "good" looks like.

    RandomModel       — floor; picks UP / DOWN / FLAT uniformly at random
    AlwaysUpModel     — buy UP, hold to resolution
    MomentumBaseline  — 30s BTC momentum, trades dynamically — THE BAR
"""

from polybench.baselines import AlwaysUpModel, MomentumBaseline, RandomModel

__all__ = ["AlwaysUpModel", "MomentumBaseline", "RandomModel"]
