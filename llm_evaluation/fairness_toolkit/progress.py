"""Progress bars and phase logging for long CHOCLO runs."""

from __future__ import annotations

import sys
from typing import Optional, Protocol


class ProgressBar(Protocol):
    def update(self, n: int = 1) -> None: ...

    def set_postfix_str(self, s: str, refresh: bool = True) -> None: ...

    def close(self) -> None: ...


class _FallbackProgress:
    """Simple text progress when tqdm is unavailable."""

    def __init__(self, total: int, desc: str) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.n = 0
        self._last_pct = -1

    def update(self, n: int = 1) -> None:
        if self.total <= 0:
            return
        self.n = min(self.n + n, self.total)
        pct = int(100 * self.n / self.total)
        if self.n == 1 or self.n == self.total or pct >= self._last_pct + 5:
            self._last_pct = pct
            print(f"  {self.desc}: {self.n}/{self.total} ({pct}%)", flush=True)

    def set_postfix_str(self, s: str, refresh: bool = True) -> None:
        if self.n == 0:
            print(f"  {self.desc} — {s}", flush=True)

    def close(self) -> None:
        if self.total > 0 and self.n >= self.total:
            print(f"  {self.desc}: completado ({self.total} tareas)", flush=True)


def log_phase(title: str) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(title, flush=True)
    print("=" * 60, flush=True)


def log_step(message: str) -> None:
    print(f"  -> {message}", flush=True)


def task_progress(
    total: int,
    desc: str,
    *,
    unit: str = "task",
    initial: int = 0,
) -> ProgressBar:
    """Create a tqdm progress bar (or a lightweight fallback)."""
    if total <= 0:
        bar = _FallbackProgress(0, desc)
        return bar

    try:
        from tqdm import tqdm

        return tqdm(
            total=total,
            desc=desc,
            unit=unit,
            initial=initial,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.5,
        )
    except ImportError:
        bar = _FallbackProgress(total, desc)
        if initial:
            bar.update(initial)
        return bar


def count_response_tasks(
    df,
    model_names,
    *,
    resume: bool,
    response_is_done_fn,
) -> tuple[int, int]:
    """Return (total_tasks, already_done)."""
    total = len(df) * len(model_names)
    done = 0
    if not resume:
        return total, 0

    for row_idx in df.index:
        for model_name in model_names:
            col = f"response_{model_name}"
            if response_is_done_fn(df.at[row_idx, col]):
                done += 1
    return total, done
