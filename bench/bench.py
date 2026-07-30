"""Reproducible fitter benchmarks against upstream lifelines."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import lifelines as upstream  # noqa: E402
import mojo_lifelines as mojo  # noqa: E402

warnings.filterwarnings("ignore")


def best_time(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def survival_data(n, seed=0, rounded=False):
    rng = np.random.default_rng(seed)
    durations = rng.exponential(12.0, n)
    if rounded:
        durations = np.ceil(durations)
    events = rng.random(n) < 0.68
    return durations, events


def cox_data(n, d, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    beta = rng.normal(scale=0.25, size=d)
    event_time = rng.exponential(2.0, n) / np.exp(x @ beta)
    censor_time = rng.exponential(4.0, n)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(d)])
    frame["T"] = np.minimum(event_time, censor_time)
    frame["E"] = event_time <= censor_time
    return frame


def main():
    cases = []

    durations, events = survival_data(500_000, rounded=True)
    cases.append(
        (
            "KaplanMeierFitter.fit (500k, tied)",
            lambda: mojo.KaplanMeierFitter().fit(durations, events),
            lambda: upstream.KaplanMeierFitter().fit(durations, events),
        )
    )
    cases.append(
        (
            "NelsonAalenFitter.fit (500k, tied)",
            lambda: mojo.NelsonAalenFitter().fit(durations, events),
            lambda: upstream.NelsonAalenFitter().fit(durations, events),
        )
    )

    param_durations, param_events = survival_data(100_000, seed=2)
    cases.append(
        (
            "WeibullFitter.fit (100k)",
            lambda: mojo.WeibullFitter().fit(param_durations, param_events),
            lambda: upstream.WeibullFitter().fit(param_durations, param_events),
        )
    )

    frame = cox_data(20_000, 8, seed=3)
    cases.append(
        (
            "CoxPHFitter.fit (20k x 8)",
            lambda: mojo.CoxPHFitter().fit(frame, "T", "E"),
            lambda: upstream.CoxPHFitter().fit(frame, "T", "E"),
        )
    )

    fit_frame = cox_data(20_000, 8, seed=4)
    predict_frame = cox_data(300_000, 8, seed=5).drop(columns=["T", "E"])
    ours_model = mojo.CoxPHFitter().fit(fit_frame, "T", "E")
    theirs_model = upstream.CoxPHFitter().fit(fit_frame, "T", "E")
    cases.append(
        (
            "Cox predict_partial_hazard (300k x 8)",
            lambda: ours_model.predict_partial_hazard(predict_frame),
            lambda: theirs_model.predict_partial_hazard(predict_frame),
        )
    )

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-lifelines | lifelines | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases:
        ours()
        theirs()
        a = best_time(ours)
        b = best_time(theirs)
        ratio = b / a
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(f"| {name} | {a * 1000:.1f} ms | {b * 1000:.1f} ms | {result} |")


if __name__ == "__main__":
    main()
