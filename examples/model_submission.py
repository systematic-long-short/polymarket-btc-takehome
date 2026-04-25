"""CANDIDATE TEMPLATE — rename / edit this file.

Your submission MUST be a single file named ``model_submission.py`` at the
top level of your submission directory, containing a class ``ModelSubmission``
that subclasses ``polybench.Model``.

Read the README for the full contract, the security-scanner allowlist, and
the scoring philosophy before you start.
"""

from __future__ import annotations

from collections import deque
from typing import Any

# `polybench` is installed via `pip install -e .` in the repo root.
from polybench import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick

# You can import the libraries listed in README.md plus the documented
# safe standard-library subset. The security scanner rejects unknown imports
# and direct file/environment/subprocess access.
#
# Example allowed imports:
#   import numpy as np
#   import pandas as pd
#   import polars as pl
#   from sklearn.linear_model import LogisticRegression
#   import lightgbm as lgb
#   import xgboost as xgb
#   import torch
#   import tensorflow as tf
#   import optuna
#   import ta
#
# Network calls in on_tick are allowed but count against the 500 ms budget.
# If you need slow external feeds, start a background thread in on_start:
#
#   import threading, httpx, queue
#
#   def on_start(self, market_info):
#       self._q = queue.Queue()
#       t = threading.Thread(target=self._puller, daemon=True)
#       t.start()
#
#   def _puller(self):
#       with httpx.Client(timeout=2.0) as client:
#           while not self._stopping:
#               resp = client.get("https://api.example.com/orderbook")
#               self._q.put(resp.json())
#               time.sleep(1.0)


class ModelSubmission(Model):
    """Your trading model.

    Required: override ``on_tick``. Return a ``Signal`` (or ``None``/``FLAT``).
    Optional: override ``on_start`` and ``on_finish``.

    State lives on ``self``. The harness instantiates your class ONCE per
    run and reuses it across every event in the run window.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        # TODO: set your hyperparameters here.
        # Example: self._lookback = 30
        self._btc_window: deque[float] = deque(maxlen=600)   # 10 min @ 1 Hz
        self._up_window: deque[float] = deque(maxlen=600)

    def on_start(self, market_info: MarketInfo) -> None:
        # Called ONCE at the start of every 5-minute event during the run.
        # Use this to reset per-event state (e.g. warm-up flags).
        # TODO: anything you'd like to initialize per event.
        pass

    def on_tick(self, tick: Tick) -> Signal | None:
        # Called at 1 Hz while the event is trading.
        # You have ~500 ms wall-clock here — the harness will drop the
        # signal if you overrun.
        #
        # Return a Signal(side, size, confidence) where:
        #   side       = Side.UP | Side.DOWN | Side.FLAT
        #   size       = fraction of starting capital to allocate ∈ [0, 1]
        #   confidence = analytics-only value ∈ [0, 1]
        #
        # Returning FLAT or None closes any open position.
        self._btc_window.append(tick.btc_last)
        self._up_window.append(tick.up_mid)

        # TODO: compute your signal here. This template returns FLAT.
        return FLAT

    def on_finish(self, result: RunResult) -> None:
        # Called once at the end of the whole run. Use for teardown/logging.
        # TODO: (optional) print summary or stop background threads.
        pass
