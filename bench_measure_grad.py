#!/usr/bin/env python3
"""Benchmark measure_grad timing — quantifies the cost of adding stomach_cm.

Compares:
- all keys EXCLUDING stomach  (= the pre-stomach baseline)
- all keys INCLUDING stomach  (= after stomach was added)
- stomach alone

on all 6 testdata bodies, reporting mean / min / std of wall-clock time
over ``N_ITERS`` gradient forward+backward passes.
"""

import os
import statistics
import time

import torch

from clad_body.load.anny import load_anny_from_params
from clad_body.measure.anny import SUPPORTED_KEYS, load_phenotype_params, measure_grad

TESTDATA_DIR = os.path.join(
    os.path.dirname(__file__), "clad_body", "measure", "testdata", "anny"
)

SUBJECTS = [
    "male_average",
    "female_average",
    "male_plus_size",
    "female_curvy",
    "female_slim",
    "female_plus_size",
]

N_WARMUP = 3
N_ITERS = 10

BASELINE_KEYS = sorted(k for k in SUPPORTED_KEYS if k != "stomach_cm")
FULL_KEYS = sorted(SUPPORTED_KEYS)


def bench_one(body, keys):
    for _ in range(N_WARMUP):
        m = measure_grad(body, only=keys)
        loss = sum(m.values())
        loss.backward()
        for t in body.phenotype_kwargs.values():
            if t.grad is not None:
                t.grad = None

    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        m = measure_grad(body, only=keys)
        loss = sum(m.values())
        loss.backward()
        for t in body.phenotype_kwargs.values():
            if t.grad is not None:
                t.grad = None
        times.append(time.perf_counter() - t0)
    return times


def main():
    print(f"SUPPORTED_KEYS ({len(SUPPORTED_KEYS)}): {FULL_KEYS}")
    print(f"Warmup={N_WARMUP}, Iters={N_ITERS}\n")

    header = f"{'subject':25s}  {'baseline (no stomach)':>23s}  {'full (w/ stomach)':>20s}  {'Δ':>10s}  {'stomach only':>14s}"
    print(header)
    print("-" * len(header))

    baseline_mins, full_mins, stomach_mins = [], [], []

    for name in SUBJECTS:
        params = load_phenotype_params(os.path.join(TESTDATA_DIR, name, "anny_params.json"))
        body = load_anny_from_params(params, requires_grad=True)

        t_base = bench_one(body, BASELINE_KEYS)
        t_full = bench_one(body, FULL_KEYS)
        t_stom = bench_one(body, ["stomach_cm"])

        mb = min(t_base) * 1000
        mf = min(t_full) * 1000
        ms = min(t_stom) * 1000
        delta = mf - mb

        baseline_mins.append(mb)
        full_mins.append(mf)
        stomach_mins.append(ms)

        print(f"{name:25s}  "
              f"{mb:8.1f} ms (mean {statistics.mean(t_base)*1000:6.1f})  "
              f"{mf:8.1f} ms (mean {statistics.mean(t_full)*1000:6.1f})  "
              f"{delta:+6.1f} ms  "
              f"{ms:8.1f} ms")

    print()
    print(f"Aggregate (min over N_ITERS, averaged over {len(SUBJECTS)} subjects):")
    print(f"  baseline (no stomach)     : {statistics.mean(baseline_mins):6.1f} ms")
    print(f"  full (with stomach)       : {statistics.mean(full_mins):6.1f} ms")
    print(f"  Δ from adding stomach     : {statistics.mean(full_mins) - statistics.mean(baseline_mins):+6.1f} ms")
    print(f"  stomach alone (isolated)  : {statistics.mean(stomach_mins):6.1f} ms")


if __name__ == "__main__":
    main()
