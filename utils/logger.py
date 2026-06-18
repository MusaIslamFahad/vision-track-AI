"""
logger.py — Weights & Biases experiment logger.
Gracefully disabled when WANDB_API_KEY is not set.
"""

from __future__ import annotations

import os
import time
from typing import Optional

_wandb_available = False
try:
    import wandb
    _wandb_available = True
except ImportError:
    pass


class WandbLogger:
    """
    Thin wrapper around wandb for logging detection sessions.
    If wandb is not configured (no API key), all methods are no-ops.
    """

    def __init__(self, project: str = "object-detection-tracker", entity: Optional[str] = None):
        self.enabled = False
        self._run = None

        api_key = os.getenv("WANDB_API_KEY", "").strip()
        if not _wandb_available or not api_key:
            return

        try:
            wandb.login(key=api_key, relogin=False)
            self.enabled = True
            self._project = project
            self._entity = entity or os.getenv("WANDB_ENTITY") or None
        except Exception:
            self.enabled = False

    def start_run(self, config: dict, run_name: Optional[str] = None) -> None:
        if not self.enabled:
            return
        try:
            self._run = wandb.init(
                project=self._project,
                entity=self._entity,
                name=run_name or f"session-{int(time.time())}",
                config=config,
                reinit=True,
            )
        except Exception:
            self.enabled = False

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            wandb.log(metrics, step=step)
        except Exception:
            pass

    def log_summary(self, stats: dict) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            for k, v in stats.items():
                wandb.run.summary[k] = v
        except Exception:
            pass

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            wandb.finish()
            self._run = None
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finish()
