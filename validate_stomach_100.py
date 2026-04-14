#!/usr/bin/env python3
"""Validate differentiable stomach_cm against measure() on 100 random bodies
from the 10k questionnaire dataset.

Reports MAE, max error, bias, and produces a scatter plot + error histogram.
Also highlights the worst cases so we can inspect them.
"""

import json
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from clad_body.load.anny import load_anny_from_params
from clad_body.measure import measure
from clad_body.measure.anny import measure_grad

DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "..", "body-tuning", "questionnaire", "data_10k_42", "test.json",
))

N_BODIES = 100
SEED = 42


def main():
    with open(DATA_PATH) as f:
        dataset = json.load(f)

    random.seed(SEED)
    idxs = random.sample(range(len(dataset)), N_BODIES)

    results = []

    for i, idx in enumerate(idxs):
        entry = dataset[idx]
        params = entry["params"]
        ref_stomach = entry["measurements"]["stomach_cm"]
        ref_stomach_z = entry["measurements"].get("_stomach_z", 0.0)
        gender = entry["measurements"]["labels"]["gender"]
        body_shape = entry["measurements"]["labels"].get("body_shape", "?")

        try:
            body = load_anny_from_params(params)
            m_grad = measure_grad(body, only=["stomach_cm"])
            soft_stomach = m_grad["stomach_cm"].item()
            err = soft_stomach - ref_stomach
            results.append({
                "idx": idx,
                "gender": gender,
                "shape": body_shape,
                "ref": ref_stomach,
                "ref_z": ref_stomach_z,
                "soft": soft_stomach,
                "err": err,
            })
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_BODIES} processed, last err={err:+.2f} cm")
        except Exception as e:
            print(f"  [{idx}] FAILED: {e}")

    errs = np.array([r["err"] for r in results])
    abs_errs = np.abs(errs)
    print(f"\n{'='*60}")
    print(f"Stomach MAE on {len(results)} bodies:")
    print(f"  MAE          = {abs_errs.mean():.3f} cm")
    print(f"  median AE    = {np.median(abs_errs):.3f} cm")
    print(f"  P90 AE       = {np.percentile(abs_errs, 90):.3f} cm")
    print(f"  P95 AE       = {np.percentile(abs_errs, 95):.3f} cm")
    print(f"  max          = {abs_errs.max():.3f} cm")
    print(f"  bias (mean)  = {errs.mean():+.3f} cm")
    print(f"  std          = {errs.std():.3f} cm")
    print(f"{'='*60}\n")

    # By gender breakdown
    for gender in ["male", "female"]:
        grp_errs = np.array([r["err"] for r in results if r["gender"] == gender])
        if len(grp_errs):
            print(f"  {gender:8s} (n={len(grp_errs):3d}): MAE={np.abs(grp_errs).mean():.2f}  "
                  f"max={np.abs(grp_errs).max():.2f}  bias={grp_errs.mean():+.2f}")

    # Worst 10
    print("\nWorst 10 cases:")
    print(f"{'idx':>5s}  {'gender':>6s}  {'shape':>16s}  {'ref':>7s}  {'soft':>7s}  {'err':>6s}")
    for r in sorted(results, key=lambda r: -abs(r["err"]))[:10]:
        print(f"  {r['idx']:>4d}  {r['gender']:>6s}  {r['shape']:>16s}  "
              f"{r['ref']:7.2f}  {r['soft']:7.2f}  {r['err']:+6.2f}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Scatter plot
    ax = axes[0]
    refs = np.array([r["ref"] for r in results])
    softs = np.array([r["soft"] for r in results])
    genders = np.array([r["gender"] for r in results])
    for g, color in [("female", "tab:red"), ("male", "tab:blue")]:
        mask = genders == g
        ax.scatter(refs[mask], softs[mask], s=30, alpha=0.7, label=g, c=color)
    lo, hi = min(refs.min(), softs.min()) - 3, max(refs.max(), softs.max()) + 3
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.plot([lo, hi], [lo + 2.5, hi + 2.5], "--", c="gray", alpha=0.5, label="±2.5 cm")
    ax.plot([lo, hi], [lo - 2.5, hi - 2.5], "--", c="gray", alpha=0.5)
    ax.set_xlabel("measure() stomach_cm (reference)", fontsize=11)
    ax.set_ylabel("measure_grad stomach_cm (soft)", fontsize=11)
    ax.set_title(f"Agreement on {len(results)} bodies\n"
                 f"MAE={abs_errs.mean():.2f} cm, max={abs_errs.max():.2f} cm",
                 fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Error histogram
    ax = axes[1]
    ax.hist(errs, bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax.axvline(0, c="k", lw=1)
    ax.axvline(errs.mean(), c="red", lw=1.5, label=f"bias = {errs.mean():+.2f}")
    ax.axvline(errs.mean() + errs.std(), c="red", lw=1, linestyle="--", label=f"±1σ = {errs.std():.2f}")
    ax.axvline(errs.mean() - errs.std(), c="red", lw=1, linestyle="--")
    ax.set_xlabel("soft − ref (cm)", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title("Error distribution", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Error vs ref stomach
    ax = axes[2]
    for g, color in [("female", "tab:red"), ("male", "tab:blue")]:
        mask = genders == g
        ax.scatter(refs[mask], errs[mask], s=30, alpha=0.7, label=g, c=color)
    ax.axhline(0, c="k", lw=1)
    ax.axhline(2.5, c="gray", lw=1, linestyle="--")
    ax.axhline(-2.5, c="gray", lw=1, linestyle="--")
    ax.set_xlabel("ref stomach_cm", fontsize=11)
    ax.set_ylabel("error = soft − ref (cm)", fontsize=11)
    ax.set_title("Error vs reference circumference", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__),
                            "viz_stomach_output", f"VALIDATION_{N_BODIES}bodies.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_path}")

    # Save raw results for later inspection
    results_path = os.path.join(os.path.dirname(__file__),
                                "viz_stomach_output",
                                f"VALIDATION_{N_BODIES}bodies_raw.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {results_path}")


if __name__ == "__main__":
    main()
