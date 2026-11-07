#!/usr/bin/env python3
"""Fit Elo-scale ratings from `tavernagent league` output and plot them per run.

Ratings come from a Plackett-Luce model of the full 8-player placements (MM algorithm,
Hunter 2004); the Elo scale is 400 * log10(strength), centred on the mean. Error bars are
bootstrap standard errors over games.

    python tools/league_elo.py runs/league/league.jsonl --out runs/league/elo.png [--anchor name]

Model names are expected as `<run>_s<step>`; anything else is plotted as a single point.
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np


def load(path):
    games = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                games.append((row["models"], row["placements"]))
    return games


def plackett_luce(games, names, iters=500, tol=1e-9):
    """MM estimate of log-strengths from full rankings."""
    index = {n: i for i, n in enumerate(names)}
    rankings = []
    for models, places in games:
        order = sorted(range(len(models)), key=lambda s: places[s])
        rankings.append([index[models[s]] for s in order])
    n = len(names)
    # Group equal table sizes; full league tables have eight distinct models.
    groups = [np.array([r for r in rankings if len(r)==k],dtype=int)
              for k in sorted({len(r) for r in rankings})]
    wins = sum((np.bincount(r[:,:-1].ravel(),minlength=n) for r in groups),np.zeros(n))
    gamma = np.ones(n)
    for _ in range(iters):
        denom = np.zeros(n)
        for r in groups:
            suffix = np.cumsum(gamma[r][:,::-1],axis=1)[:,::-1]
            inv=1.0/suffix
            inv[:,-1]=0
            exposure=np.cumsum(inv,axis=1)
            denom+=np.bincount(r.ravel(),weights=exposure.ravel(),minlength=n)
        # A model never placed above anyone (or absent from a bootstrap sample) gets a
        # small floor instead of zero so the geometric mean stays defined.
        new = np.where(denom > 0, np.maximum(wins, 0.5) / np.maximum(denom, 1e-12), gamma)
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(gamma))) < tol:
            gamma = new
            break
        gamma = new
    return 400.0 * np.log10(gamma)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("league")
    ap.add_argument("--out", default=None, help="PNG path (default: next to the input)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--anchor", default=None, help="model whose rating is fixed at 0")
    ap.add_argument("--json", default=None, help="also write ratings as JSON")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="drop models whose name starts with this (repeatable)",
    )
    args = ap.parse_args()

    games = load(args.league)
    if args.exclude:
        # Drop the excluded models' seats; the remaining models' relative order is unchanged.
        kept = []
        for models, places in games:
            keep = [
                i for i, m in enumerate(models) if not any(m.startswith(e) for e in args.exclude)
            ]
            if len(keep) >= 2:
                kept.append(([models[i] for i in keep], [places[i] for i in keep]))
        games = kept
    names = sorted({m for models, _ in games for m in models})
    counts = defaultdict(int)
    for models, _ in games:
        for m in models:
            counts[m] += 1
    elo = plackett_luce(games, names)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.bootstrap):
        sample = [games[i] for i in rng.integers(0, len(games), len(games))]
        boots.append(plackett_luce(sample, names, iters=200))
    se = np.std(np.array(boots), axis=0)
    if args.anchor:
        elo = elo - elo[names.index(args.anchor)]

    table = sorted(zip(names, elo, se), key=lambda t: -t[1])
    print(
        f"{len(games)} games, {len(names)} models, {min(counts.values())}-{max(counts.values())} games each"
    )
    print(f"{'model':<22} {'elo':>8} {'se':>6} {'games':>6}")
    for name, e, s in table:
        print(f"{name:<22} {e:8.1f} {s:6.1f} {counts[name]:6d}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {n: {"elo": float(e), "se": float(s), "games": counts[n]} for n, e, s in table},
                handle,
                indent=1,
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = defaultdict(list)
    singles = []
    for name, e, s in zip(names, elo, se):
        m = re.fullmatch(r"(.+)_s(\d+)", name)
        if m:
            runs[m.group(1)].append((int(m.group(2)), e, s))
        else:
            singles.append((name, e, s))
    fig, ax = plt.subplots(figsize=(11, 6.5))
    # Distinct colour / marker / dash combinations so more than ten runs stay readable.
    colours = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
    dashes = ["-", "--", "-.", ":"]
    for i, (run, points) in enumerate(sorted(runs.items())):
        points.sort()
        xs = [p[0] for p in points]
        ax.errorbar(
            xs,
            [p[1] for p in points],
            yerr=[p[2] for p in points],
            color=colours[i % len(colours)],
            marker=markers[i % len(markers)],
            linestyle=dashes[i % len(dashes)],
            markersize=6,
            capsize=3,
            label=run,
        )
    for name, e, s in singles:
        ax.axhline(e, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1] if runs else 0, e, name, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("training round")
    ax.set_ylabel("Elo (Plackett-Luce, mean 0)" + (f", {args.anchor} = 0" if args.anchor else ""))
    ax.set_title(
        f"League of {len(names)} checkpoints, {len(games)} games, {min(counts.values())}-{max(counts.values())} per model"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    out = args.out or args.league.rsplit(".", 1)[0] + ".elo.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
