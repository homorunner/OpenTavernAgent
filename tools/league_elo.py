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


def encode_rankings(games, names):
    """Encode each full game once, retaining its position for bootstrap weights."""
    index = {name: i for i, name in enumerate(names)}
    groups = defaultdict(list)
    positions = defaultdict(list)
    for game_id, (models, places) in enumerate(games):
        order = sorted(range(len(models)), key=lambda seat: places[seat])
        groups[len(models)].append([index[models[seat]] for seat in order])
        positions[len(models)].append(game_id)
    return [(np.asarray(groups[k], dtype=np.int32), np.asarray(positions[k], dtype=np.intp))
            for k in sorted(groups)]


def fit_rankings(groups, n, iters=500, tol=1e-9, weights=None, initial=None):
    """Same MM estimator; integer game weights exactly represent resampling."""
    wins = np.zeros(n)
    appearances = np.zeros(n)
    prepared = []
    for rankings, positions in groups:
        w = None if weights is None else np.asarray(weights[positions], dtype=float)
        if w is not None:
            keep = w > 0
            rankings, w = rankings[keep], w[keep]
        flat = rankings.ravel()
        k = rankings.shape[1]
        wins += np.bincount(rankings[:, :-1].ravel(),
                            weights=None if w is None else np.repeat(w, k-1), minlength=n)
        appearances += np.bincount(flat, weights=None if w is None else np.repeat(w, k), minlength=n)
        prepared.append((rankings, flat, w))
    # Absent models are unidentifiable; use the original cold-start convention.
    gamma = np.ones(n) if initial is None or np.any(appearances == 0) else np.exp(
        (np.asarray(initial) - np.mean(initial)) * np.log(10) / 400)
    for _ in range(iters):
        denom = np.zeros(n)
        for rankings, flat, w in prepared:
            suffix = np.cumsum(gamma[rankings][:, ::-1], axis=1)[:, ::-1]
            inv = 1.0 / suffix
            inv[:, -1] = 0
            exposure = np.cumsum(inv, axis=1)
            if w is not None:
                exposure *= w[:, None]
            denom += np.bincount(flat, weights=exposure.ravel(), minlength=n)
        new = np.where(denom > 0, np.maximum(wins, 0.5) / np.maximum(denom, 1e-12), gamma)
        new /= np.exp(np.mean(np.log(new)))
        if np.max(np.abs(np.log(new) - np.log(gamma))) < tol:
            gamma = new
            break
        gamma = new
    return 400.0 * np.log10(gamma)


def plackett_luce(games, names, iters=500, tol=1e-9):
    """MM estimate of log-strengths from full rankings."""
    return fit_rankings(encode_rankings(games, names), len(names), iters, tol)


def ratings_with_bootstrap(games, names, bootstrap=32, workers=4, seed=0,
                           anchor=None, anchor_rating=1000.0):
    """Full-game bootstrap without rebuilding and sorting Python game lists."""
    from concurrent.futures import ThreadPoolExecutor
    groups = encode_rankings(games, names)
    elo = fit_rankings(groups, len(names))
    rng = np.random.default_rng(seed)
    # Keep the original multinomial bootstrap and deterministic sample order.
    weights = [np.bincount(rng.integers(len(games), size=len(games)), minlength=len(games))
               for _ in range(bootstrap)]
    def fit(w):
        return fit_rankings(groups, len(names), iters=100, weights=w, initial=elo)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, bootstrap))) as pool:
        boots = np.asarray(list(pool.map(fit, weights)))
    if anchor is not None:
        index = names.index(anchor)
        elo = elo - elo[index] + anchor_rating
        # Anchor each resampled fit too, preserving covariance in relative errors.
        boots = boots - boots[:, index:index+1] + anchor_rating
    counts = sum((np.bincount(r.ravel(), minlength=len(names)) for r, _ in groups), np.zeros(len(names), dtype=int))
    return elo, np.std(boots, axis=0, ddof=1), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("league")
    ap.add_argument("--out", default=None, help="PNG path (default: next to the input)")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--anchor", default=None, help="model whose rating is fixed at --anchor-rating")
    ap.add_argument("--anchor-rating", type=float, default=1000.0)
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
    if args.bootstrap < 2:
        ap.error("--bootstrap must be at least 2")
    if args.anchor and args.anchor not in names:
        ap.error("--anchor must name a model in the league")
    elo, se, _ = ratings_with_bootstrap(
        games, names, bootstrap=args.bootstrap,
        anchor=args.anchor, anchor_rating=args.anchor_rating,
    )

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
    origin = f"{args.anchor} = {args.anchor_rating:g}" if args.anchor else "pool mean = 0"
    ax.set_ylabel(f"Elo (Plackett-Luce, {origin})")
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
