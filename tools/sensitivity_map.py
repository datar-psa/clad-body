#!/usr/bin/env python3
"""Sensitivity Jacobian: |d(measurement)/d(param)| averaged across 6 testdata bodies.

For every supported measurement key, compute the gradient of that measurement
w.r.t. every phenotype tensor and every local_changes tensor on each of the 6
testdata bodies, then average the absolute values across bodies.

Output: markdown with three sections, plus two heatmap PNGs when --output is
used. PNGs land in the clad-body `assets/` directory by default (public,
referenced by the clad-body README); override with --assets-dir.
  1. Global phenotypes (11 rows)  -- the top-level levers (height, weight, ...)
  2. Local changes (~65 rows)     -- fine-grained blendshape levers
  3. Top levers per measurement   -- "for bust, the 5 most impactful params are ..."

Units: measurement-unit per unit-param (cm/unit or kg/unit). Phenotypes are
in [0, 1]; local_changes are in [-1, 1]. A cell value of 3.0 means pushing
the param by 0.1 moves the measurement by ~0.3 cm (or kg).

Uses:
  - Picking what to freeze/unfreeze when fitting a target measurement.
  - Debugging stuck optimisation ("bust won't move -- what's its lever?").
  - Documenting Anny's behaviour for future contributors.

Caveat: sensitivity is LOCAL. A plus-size body has higher |d(bust)/d(weight)|
than a slim one. Averaging over 6 bodies gives typical leverage, not universal.

Usage:
    cd hmr/clad-body
    venv/bin/python tools/sensitivity_map.py                           # stdout, no PNG
    venv/bin/python tools/sensitivity_map.py --output PATH             # MD + PNGs
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import date

from clad_body.load.anny import load_anny_from_params
from clad_body.measure.anny import (
    SUPPORTED_KEYS,
    load_phenotype_params,
    measure_grad,
)

TESTDATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "clad_body", "measure", "testdata", "anny",
)

SUBJECTS = [
    "male_average",
    "female_average",
    "male_plus_size",
    "female_curvy",
    "female_slim",
    "female_plus_size",
]

# Measurements as columns. Scalars/lengths first, then circumferences
# (torso -> abdomen -> legs -> arms) for readable grouping.
KEY_ORDER = [
    "height_cm",
    "inseam_cm",
    "sleeve_length_cm",
    "mass_kg",
    "bust_cm",
    "underbust_cm",
    "waist_cm",
    "stomach_cm",
    "hip_cm",
    "thigh_cm",
    "upperarm_cm",
]


def _load(subject):
    path = os.path.join(TESTDATA_DIR, subject, "anny_params.json")
    params = load_phenotype_params(path)
    return load_anny_from_params(params, requires_grad=True)


def _collect_abs_grads(body, key):
    """Run backward on measure_grad(body)[key] and return {(type, label): |grad|}."""
    for t in body.phenotype_kwargs.values():
        if t.grad is not None:
            t.grad.zero_()
    if body.local_changes_kwargs:
        for t in body.local_changes_kwargs.values():
            if t.grad is not None:
                t.grad.zero_()

    value = measure_grad(body, only=[key])[key]
    value.backward()

    out = {}
    for label, t in body.phenotype_kwargs.items():
        if t.grad is not None:
            out[("phenotype", label)] = float(t.grad.abs().item())
    if body.local_changes_kwargs:
        for label, t in body.local_changes_kwargs.items():
            if t.grad is not None:
                out[("local_changes", label)] = float(t.grad.abs().item())
    return out


def compute_jacobian():
    """Average |d(measurement)/d(param)| across the 6 testdata bodies.

    Returns:
        avg: {(type, label): {measurement_key: mean |grad|}}
        presence: {(type, label): int}  -- how many of 6 bodies had this param
    """
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    presence = defaultdict(int)

    for subject in SUBJECTS:
        print(f"  {subject} ...", file=sys.stderr)
        body = _load(subject)

        for label in body.phenotype_kwargs:
            presence[("phenotype", label)] += 1
        if body.local_changes_kwargs:
            for label in body.local_changes_kwargs:
                presence[("local_changes", label)] += 1

        for key in SUPPORTED_KEYS:
            for plabel, g in _collect_abs_grads(body, key).items():
                sums[plabel][key] += g
                counts[plabel][key] += 1

    avg = {}
    for plabel, key_sums in sums.items():
        avg[plabel] = {k: key_sums[k] / counts[plabel][k] for k in key_sums}
    return avg, presence


def _row_total(avg_row, keys):
    # Normalise each column by that measurement's max across all params, then sum.
    # Without normalisation, keys with larger absolute gradient magnitudes
    # (e.g. mass_kg, height_cm) would dominate the sort and bury levers that
    # matter specifically for circumferences. Normalised, the total is
    # "share-of-leverage" summed across measurements.
    return sum(avg_row.get(k, 0) for k in keys)


def _rank_rows(avg, keys):
    """Sort params by normalised total leverage across the given measurement keys."""
    # Column-wise max for normalisation.
    col_max = {k: max((row.get(k, 0) for row in avg.values()), default=1.0) or 1.0
               for k in keys}
    totals = []
    for plabel, row in avg.items():
        norm_total = sum(row.get(k, 0) / col_max[k] for k in keys)
        totals.append((plabel, norm_total))
    totals.sort(key=lambda t: -t[1])
    return totals


def render_heatmaps(avg, out_dir):
    """Write two PNG heatmaps to out_dir, return (phenotypes_path, local_changes_path).

    Phenotype heatmap is small enough to annotate with cell values. The
    local-changes heatmap has 70+ rows, so cell annotations are unreadable
    -- colour alone carries the signal there.

    Colour scale is log, so the 4-order-of-magnitude spread between `age`
    (~160) and `caucasian` (~0.2) is legible. Linear would render everything
    below the top-5 cells as indistinguishable yellow.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs(out_dir, exist_ok=True)

    keys = [k for k in KEY_ORDER if any(k in row for row in avg.values())]
    ranking = _rank_rows(avg, keys)

    pheno_rows = [r for r in ranking if r[0][0] == "phenotype"]
    lc_rows = [r for r in ranking if r[0][0] == "local_changes"]

    # Small positive floor so LogNorm doesn't choke on exact zeros. Cells at
    # the floor render at the lightest colour, so "near zero" and "exact zero"
    # are visually equivalent -- which is what we want here.
    FLOOR = 1e-2

    def _matrix(rows):
        M = np.full((len(rows), len(keys)), FLOOR)
        labels = []
        for i, ((ptype, label), _) in enumerate(rows):
            for j, k in enumerate(keys):
                v = avg[(ptype, label)].get(k, 0.0)
                M[i, j] = max(v, FLOOR)
            labels.append(label)
        return M, labels

    # ----- Phenotype heatmap: small, annotated ---------------------------
    M, labels = _matrix(pheno_rows)
    vmax = float(M.max())
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd",
                   norm=LogNorm(vmin=FLOOR, vmax=vmax))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate only cells above a readable threshold. Text colour switches at
    # roughly the visual midpoint of the log scale so labels stay legible on
    # both light and dark cells.
    log_vmax = np.log10(vmax)
    log_vmin = np.log10(FLOOR)
    log_mid = log_vmin + 0.6 * (log_vmax - log_vmin)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if v < 0.5:
                continue
            colour = "white" if np.log10(v) > log_mid else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color=colour)

    ax.set_title("Phenotype sensitivity  |d(measurement)/d(param)|\n"
                 "log colour scale, averaged over 6 testdata bodies", fontsize=11)
    cbar = plt.colorbar(im, ax=ax, label="|grad|  (cm or kg per unit param)")
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    pheno_path = os.path.join(out_dir, "sensitivity_phenotypes.png")
    plt.savefig(pheno_path, dpi=120)
    plt.close(fig)

    # ----- Local_changes heatmap: tall, no annotations -------------------
    M, labels = _matrix(lc_rows)
    vmax = float(M.max())
    height = max(8.0, len(labels) * 0.22)
    fig, ax = plt.subplots(figsize=(11, height))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd",
                   norm=LogNorm(vmin=FLOOR, vmax=vmax))
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Local-changes sensitivity  |d(measurement)/d(param)|\n"
                 "sorted by total normalised leverage (top = most impactful)",
                 fontsize=11)
    cbar = plt.colorbar(im, ax=ax, label="|grad|  (cm or kg per unit param)")
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    lc_path = os.path.join(out_dir, "sensitivity_local_changes.png")
    plt.savefig(lc_path, dpi=120)
    plt.close(fig)

    return pheno_path, lc_path


def format_markdown(avg, presence, *, image_rel_paths=None):
    keys = [k for k in KEY_ORDER if any(k in row for row in avg.values())]
    ranking = _rank_rows(avg, keys)

    lines = []
    lines.append("# Phenotype sensitivity of measurements")
    lines.append("")
    lines.append(f"Generated: {date.today().isoformat()}")
    lines.append(f"Bodies: {len(SUBJECTS)} ({', '.join(SUBJECTS)})")
    lines.append("")

    if image_rel_paths is not None:
        pheno_rel, lc_rel = image_rel_paths
        lines.append("## Visual summary")
        lines.append("")
        lines.append(f"![Phenotype sensitivity heatmap]({pheno_rel})")
        lines.append("")
        lines.append(f"![Local-changes sensitivity heatmap]({lc_rel})")
        lines.append("")

    lines.append("## What this is")
    lines.append("")
    lines.append("For every (parameter, measurement) pair, the absolute gradient")
    lines.append("`|d(measurement)/d(param)|` averaged across the 6 testdata bodies.")
    lines.append("Units: measurement-unit per unit-param (cm/unit, kg/unit).")
    lines.append("Phenotypes are in [0, 1]; local_changes are in [-1, 1]. A cell")
    lines.append("value of 3.0 means pushing the param by 0.1 moves the measurement")
    lines.append("by ~0.3 cm. Sign is not shown; positive and negative leverage are")
    lines.append("both \"high leverage\" for fitting purposes.")
    lines.append("")
    lines.append("Rows are sorted by **total normalised leverage**: each column is")
    lines.append("divided by its max before summing, so a row that tops one column")
    lines.append("ranks as highly as a row that is broadly mid-tier. Raw magnitudes")
    lines.append("would let `height_cm` and `mass_kg` dominate simply because they")
    lines.append("carry larger absolute gradients than circumferences.")
    lines.append("")
    lines.append("## Uses")
    lines.append("")
    lines.append("- **Fit debugging**: \"bust won't move\" -- check which params have")
    lines.append("  leverage over `bust_cm` and confirm you are unfreezing them.")
    lines.append("- **Freeze decisions**: low-total-leverage rows are safe to freeze")
    lines.append("  without losing fit quality on any measurement.")
    lines.append("- **Regularisation priors**: high-leverage params need tighter")
    lines.append("  priors if the measurement target is noisy.")
    lines.append("")
    lines.append("## Caveat")
    lines.append("")
    lines.append("Sensitivity is LOCAL to the evaluation point. A plus-size body has")
    lines.append("higher `|d(bust)/d(weight)|` than a slim one. Averaging over 6")
    lines.append("bodies gives typical leverage, not universal. Use at your own risk")
    lines.append("for bodies far from the testdata distribution.")
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("cd hmr/clad-body")
    lines.append("venv/bin/python tools/sensitivity_map.py \\")
    lines.append("    --output ../findings/sensitivity_map.md")
    lines.append("```")
    lines.append("")
    lines.append("PNGs land in `clad-body/assets/` by default (public, shared with the")
    lines.append("clad-body README). Override with `--assets-dir PATH`.")
    lines.append("")

    pheno_rows = [r for r in ranking if r[0][0] == "phenotype"]
    lc_rows = [r for r in ranking if r[0][0] == "local_changes"]

    def emit_table(rows, title):
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("_No rows._")
            lines.append("")
            return
        header = "| Param | " + " | ".join(keys) + " |"
        sep = "|---|" + "---|" * len(keys)
        lines.append(header)
        lines.append(sep)
        for (ptype, label), _score in rows:
            row = avg[(ptype, label)]
            cells = [f"{row.get(k, 0):.2f}" for k in keys]
            count = presence[(ptype, label)]
            suffix = "" if count == len(SUBJECTS) else f" ({count}/{len(SUBJECTS)})"
            lines.append(f"| `{label}`{suffix} | " + " | ".join(cells) + " |")
        lines.append("")

    emit_table(pheno_rows, f"Global phenotypes ({len(pheno_rows)} params)")
    emit_table(lc_rows, f"Local changes ({len(lc_rows)} params, sorted by total normalised leverage)")

    lines.append("## Top 5 levers per measurement")
    lines.append("")
    lines.append("For each measurement, the 5 params with the highest absolute")
    lines.append("average gradient. Quick lookup when planning what to unfreeze")
    lines.append("for a specific fit target.")
    lines.append("")
    for k in keys:
        ranked = [(plabel, avg[plabel].get(k, 0)) for plabel in avg if k in avg[plabel]]
        ranked.sort(key=lambda t: -t[1])
        top = ranked[:5]
        lines.append(f"### `{k}`")
        lines.append("")
        for (ptype, label), v in top:
            tag = "phenotype" if ptype == "phenotype" else "local"
            lines.append(f"- `{label}` ({tag}): **{v:.3f}**")
        lines.append("")

    return "\n".join(lines)


def _default_assets_dir():
    """clad-body/assets/ resolved from this script's location (tools/../assets)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=None,
                        help="Write markdown to this path (default: stdout, no PNGs)")
    parser.add_argument("--assets-dir", default=None,
                        help=f"Where to write heatmap PNGs "
                             f"(default: clad-body/assets/ at {_default_assets_dir()}). "
                             f"Only used with --output.")
    args = parser.parse_args()

    print("Computing Jacobian across 6 testdata bodies...", file=sys.stderr)
    avg, presence = compute_jacobian()

    if args.output:
        out_path = os.path.abspath(args.output)
        out_dir = os.path.dirname(out_path)
        assets_dir = os.path.abspath(args.assets_dir) if args.assets_dir \
            else _default_assets_dir()

        print("Rendering heatmaps...", file=sys.stderr)
        pheno_path, lc_path = render_heatmaps(avg, assets_dir)
        pheno_rel = os.path.relpath(pheno_path, start=out_dir)
        lc_rel = os.path.relpath(lc_path, start=out_dir)

        md = format_markdown(avg, presence, image_rel_paths=(pheno_rel, lc_rel))
        with open(out_path, "w") as f:
            f.write(md)
        print(f"wrote {out_path}", file=sys.stderr)
        print(f"wrote {pheno_path}", file=sys.stderr)
        print(f"wrote {lc_path}", file=sys.stderr)
    else:
        md = format_markdown(avg, presence)
        print(md)


if __name__ == "__main__":
    main()
