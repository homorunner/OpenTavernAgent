"""Self-play: the whole table plays the network, then we train on it.

Per iteration: `--games` games in which **all eight seats** use the current weights
with a small PUCT search, and every seat's decisions are recorded. Nothing is played
against the scripted bot, so the data distribution is the network's own. After the
games, `--epochs` passes over that batch and a checkpoint is written.

    python -m tavernagent.selfplay --steps 20 --games 100 --sims 8 --workers 12

Training is AdamW with a cosine schedule, gradient clipping and an EMA copy; the EMA
weights are what `eval.py` should be pointed at.

Games run in worker processes (`--workers`), each loading the latest weights from
disk. Collection time depends on simulation count, game length and worker count.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass

# One BLAS thread per worker avoids CPU oversubscription.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import taverngym

from . import mcts
from .heroes import select_heroes
from .net import N_ACTIONS, PolicyValueNet
from .rules import FREEZE_CAP, MOVE_CAP, add_budget_args, experiment_settings, game_config
from .worker import worker_call

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINER_FORMAT = 2


@dataclass
class Sample:
    x: np.ndarray
    cards: np.ndarray
    mask: np.ndarray
    pi: np.ndarray
    z: float = 0.0
    action: int = -1


@dataclass
class Stats:
    games: int = 0
    decisions: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0


# ---------------------------------------------------------------- data


def collect_chunk(args) -> tuple[list[Sample], list[float], dict]:
    """Worker: play `n` games with the weights in `path` and return the samples.

    A game with all eight seats yields ~1200 decisions, so the worker subsamples down
    to `cap` before returning: 100 games would otherwise carry 1.5 GB of features.
    """
    cpu_start = time.process_time()
    path, seeds, sims, action_cap, horizon, temperature, cap, noise, search_options = args
    net = worker_call(PolicyValueNet.load, f"checkpoint={path}", path)
    out: list[Sample] = []
    placements: list[float] = []
    for seed in seeds:
        s, place = worker_call(
            play_one,
            f"self-play seed={seed}",
            net,
            sims,
            seed,
            action_cap,
            horizon,
            temperature,
            noise,
            **search_options,
        )
        out.extend(s)
        placements.append(place)
    decisions = len(out)
    if cap and len(out) > cap:
        keep = np.random.default_rng(np.random.SeedSequence([seeds[0], 0x434150])).choice(
            len(out), cap, replace=False
        )
        out = [out[i] for i in keep]
    return out, placements, {"cpu_seconds": time.process_time() - cpu_start, "decisions": decisions}


def play_one(
    net,
    sims,
    seed,
    action_cap,
    horizon,
    temperature,
    noise=0.0,
    noise_alpha=0.3,
    c_puct=1.5,
    move_cap=MOVE_CAP,
    freeze_cap=FREEZE_CAP,
) -> tuple[list[Sample], float]:
    config = game_config(action_cap, move_cap, freeze_cap)
    game = taverngym.Game(seed=seed, config_json=config)
    select_heroes(game, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x504C4159]))
    samples: list[Sample] = []
    seats: list[int] = []
    while not game.is_over():
        acted = False
        for seat in range(game.num_players):
            if not game.needs_action(seat):
                continue
            acted = True
            x, cards, mask = mcts.observe(game, seat)
            result = mcts.search(
                game,
                seat,
                net,
                sims=sims,
                horizon=horizon,
                root_noise=noise,
                noise_alpha=noise_alpha,
                c_puct=c_puct,
                rng=rng,
            )
            action = result.best_action if temperature <= 0 else _sample(result, temperature, rng)
            samples.append(
                Sample(
                    np.array(x, copy=True), cards.copy(), mask.copy(), result.policy, action=action
                )
            )
            seats.append(seat)
            # plain step: every seat is the network here, so nothing may drive the bots
            game.step(seat, action)
        if not acted:
            raise RuntimeError(f"self-play stalled before game end: seed={seed}, turn={game.turn}")
    places = list(game.placements())
    for s, seat in zip(samples, seats):
        s.z = mcts.placement_reward(places[seat])
    return samples, float(mcts.placement_reward(places[0]))


def _sample(result, temperature, rng):
    # Work in log space so low temperatures cannot underflow all visit weights.
    policy = result.policy.astype(np.float64)
    legal = policy > 0
    weights = np.zeros_like(policy)
    logits = np.log(policy[legal]) / temperature
    weights[legal] = np.exp(logits - logits.max())
    return int(rng.choice(N_ACTIONS, p=weights / weights.sum()))


def cosine_lr(initial, minimum, index, total):
    progress = min(1.0, index / max(1, total - 1))
    return minimum + 0.5 * (initial - minimum) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------- training


class Trainer:
    """AdamW + gradient clipping on the residual net."""

    def __init__(
        self,
        net: PolicyValueNet,
        weight_decay: float = 0.01,
        value_weight: float = 0.5,
        clip: float = 5.0,
        ema_decay: float = 0.99,
        value_samples: str = "all",
    ):
        if value_samples not in ("all", "end"):
            raise ValueError("value_samples must be all or end")
        self.net = net
        self.wd = weight_decay
        self.value_weight = value_weight
        self.clip = clip
        self.ema_decay = ema_decay
        self.value_samples = value_samples
        self.last_metrics = {}
        self.m = {k: np.zeros_like(v) for k, v in net.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in net.p.items()}
        self.t = 0
        self.ema = {k: v.copy() for k, v in net.p.items()}

    def step(
        self,
        xs,
        cards,
        masks,
        pis,
        zs,
        lr: float,
        actions=None,
        pg_weight: float = 0.0,
        smoothing: float = 0.0,
    ):
        net = self.net
        logits, value = net.forward(xs, cards)
        masked = np.where(masks, logits, np.float32(-1e9))
        probs = _softmax(masked)
        b = len(xs)
        if self.value_samples == "end" and actions is None:
            raise ValueError("end-only value training requires selected actions")
        acts = np.full(b, -1, np.int64) if actions is None else np.asarray(actions, np.int64)
        if acts.shape != (b,) or np.any((acts < -1) | (acts >= N_ACTIONS)):
            raise ValueError("expected selected actions, with -1 for unknown")
        valid = acts >= 0
        if not np.all(masks[np.flatnonzero(valid), acts[valid]]):
            raise ValueError("selected training actions must be legal")
        if smoothing:
            legal = masks.astype(np.float32)
            uniform = legal / np.maximum(legal.sum(axis=1, keepdims=True), 1.0)
            pis = (1.0 - smoothing) * pis + smoothing * uniform
        dlogits = (probs - pis) / b
        if pg_weight and actions is not None:
            # REINFORCE uses the value prediction as its baseline.
            advantage = zs - value
            onehot = np.zeros_like(probs)
            onehot[np.flatnonzero(valid), acts[valid]] = 1.0
            dlogits = dlogits + pg_weight * (advantage * valid)[:, None] * (probs - onehot) / b
        dlogits = dlogits.astype(np.float32)
        # End-mode value labels come only from selected EndTurn actions.
        value_mask = np.ones(b, bool) if self.value_samples == "all" else acts == 0
        n_values = int(value_mask.sum())
        dv = (
            self.value_weight
            * 2.0
            * (value - zs)
            * (1.0 - value * value)
            * value_mask
            / max(1, n_values)
        )[:, None]
        value_loss = float(np.mean((value[value_mask] - zs[value_mask]) ** 2)) if n_values else 0.0
        policy_loss = float(-np.mean(np.sum(pis * np.log(np.maximum(probs, 1e-12)), axis=1)))
        g = net.backward(dlogits, dv.astype(np.float32))

        norm = math.sqrt(sum(float((grad * grad).sum()) for grad in g.values()))
        if not math.isfinite(norm) or not math.isfinite(policy_loss + value_loss):
            raise FloatingPointError("nonfinite training loss or gradient")
        scale = self.clip / norm if norm > self.clip else 1.0
        log_probs = np.log(np.maximum(probs, 1e-12))
        target_log = np.log(np.maximum(pis, 1e-12))
        legal_logits = np.where(masks, logits, np.nan)
        self.last_metrics = {
            "grad_norm": norm,
            "policy_head_grad_norm": math.sqrt(
                sum(float((g[k] * g[k]).sum()) for k in ("Wp", "bp"))
            ),
            "value_head_grad_norm": math.sqrt(
                sum(float((g[k] * g[k]).sum()) for k in ("Wv", "bv"))
            ),
            "value_training_samples": n_values,
            "clipped_fraction": float(scale < 1),
            "policy_entropy": float(-np.mean(np.sum(probs * log_probs, axis=1))),
            "target_entropy": float(-np.mean(np.sum(pis * target_log, axis=1))),
            "target_kl": float(np.mean(np.sum(pis * (target_log - log_probs), axis=1))),
            "legal_logit_span": float(
                np.mean(np.nanmax(legal_logits, axis=1) - np.nanmin(legal_logits, axis=1))
            ),
            "value_mean": float(value.mean()),
            "value_std": float(value.std()),
            "target_value_mean": float(zs.mean()),
            "advantage_std": float((zs - value).std()),
            "policy_gradient_norm": float(np.linalg.norm(dlogits)),
        }

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, grad in g.items():
            grad = grad * np.float32(scale)
            self.m[k] = b1 * self.m[k] + (1 - b1) * grad
            self.v[k] = b2 * self.v[k] + (1 - b2) * grad * grad
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            update = (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
            if k.startswith("W"):
                update = update + np.float32(lr * self.wd) * net.p[k]
            net.p[k] = (net.p[k] - update).astype(np.float32)
            self.ema[k] = (self.ema_decay * self.ema[k] + (1 - self.ema_decay) * net.p[k]).astype(
                np.float32
            )
        return policy_loss, value_loss


def _softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def train_epoch(trainer, samples, batch_size=128, lr=3e-4, pg_weight=0.0, smoothing=0.0, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    order = rng.permutation(len(samples))
    pl = vl = 0.0
    batches = 0
    metrics = []
    for start in range(0, len(samples), batch_size):
        idx = order[start : start + batch_size]
        xs = np.stack([samples[i].x for i in idx])
        cards = np.stack([samples[i].cards for i in idx])
        masks = np.stack([samples[i].mask for i in idx])
        pis = np.stack([samples[i].pi for i in idx])
        zs = np.array([samples[i].z for i in idx], dtype=np.float32)
        acts = np.array([samples[i].action for i in idx], dtype=np.int64)
        a, b = trainer.step(
            xs, cards, masks, pis, zs, lr, actions=acts, pg_weight=pg_weight, smoothing=smoothing
        )
        metrics.append(trainer.last_metrics.copy())
        pl += a
        vl += b
        batches += 1
    trainer.epoch_metrics = (
        {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}
    )
    return pl / max(batches, 1), vl / max(batches, 1)


class ReplayWindow:
    """Equal-sized per-round batches; uniformly sample recent transitions."""

    def __init__(self, rounds=1):
        if rounds < 1:
            raise ValueError("replay window must be positive")
        self.rounds = deque(maxlen=rounds)

    def add_and_sample(self, samples, limit, rng):
        self.rounds.append(samples)
        if len(self.rounds) == 1:
            return samples
        pool = [s for batch in self.rounds for s in batch]
        keep = rng.choice(len(pool), min(limit, len(pool)), replace=False)
        return [pool[i] for i in keep]


def save_training_state(path, trainer, step, train_rng, replay_rng, history, stats, options):
    """Atomic, pickle-free optimizer/RNG snapshot for exact continued training."""
    path = pathlib.Path(path)
    meta = {
        "format": TRAINER_FORMAT,
        "search_version": mcts.SEARCH_VERSION,
        "step": step,
        "optimizer_step": trainer.t,
        "optimizer_hparams": {
            "wd": trainer.wd,
            "clip": trainer.clip,
            "value_weight": trainer.value_weight,
            "ema_decay": trainer.ema_decay,
            "value_samples": trainer.value_samples,
        },
        "card_fingerprint": trainer.net.card_fingerprint,
        "embedding_dim": trainer.net.embedding_dim,
        "activation": trainer.net.activation,
        "residual_scale": trainer.net.residual_scale,
        "head_norm": trainer.net.head_norm,
        "train_rng": train_rng.bit_generator.state,
        "replay_rng": replay_rng.bit_generator.state,
        "history": history,
        "stats": vars(stats),
        "options": vars(options),
    }
    arrays = {
        f"{prefix}/{key}": value
        for prefix, parameters in (
            ("net", trainer.net.p),
            ("m", trainer.m),
            ("v", trainer.v),
            ("ema", trainer.ema),
        )
        for key, value in parameters.items()
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays, metadata=json.dumps(meta))
    temporary.replace(path)


def load_training_state(path, trainer, train_rng, replay_rng):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        required = {
            "format",
            "search_version",
            "optimizer_step",
            "optimizer_hparams",
            "card_fingerprint",
            "embedding_dim",
            "activation",
            "residual_scale",
            "head_norm",
            "train_rng",
            "replay_rng",
            "step",
            "history",
            "stats",
            "options",
        }
        if not required <= meta.keys():
            raise ValueError("incomplete trainer format v2 metadata")
        hparams = meta["optimizer_hparams"]
        if not {"wd", "clip", "value_weight", "ema_decay", "value_samples"} <= hparams.keys():
            raise ValueError("incomplete trainer format v2 optimizer settings")
        value_samples = hparams["value_samples"]
        if (
            meta["format"] != TRAINER_FORMAT
            or value_samples not in ("all", "end")
            or meta["search_version"] != mcts.SEARCH_VERSION
            or meta["card_fingerprint"] != trainer.net.card_fingerprint
            or meta["embedding_dim"] != trainer.net.embedding_dim
            or meta["activation"] != trainer.net.activation
            or meta["residual_scale"] != trainer.net.residual_scale
            or meta["head_norm"] != trainer.net.head_norm
        ):
            raise ValueError("trainer snapshot does not match network schema")
        if meta["options"]["replay_rounds"] != 1:
            raise ValueError("exact replay-window restoration is not supported")
        for prefix, target in (
            ("net", trainer.net.p),
            ("m", trainer.m),
            ("v", trainer.v),
            ("ema", trainer.ema),
        ):
            for key in target:
                array = data[f"{prefix}/{key}"]
                if array.shape != target[key].shape:
                    raise ValueError(f"trainer parameter shape mismatch: {prefix}/{key}")
                target[key] = array.copy()
        trainer.t = meta["optimizer_step"]
        trainer.value_samples = value_samples
        for key in ("wd", "clip", "value_weight", "ema_decay"):
            setattr(trainer, key, hparams[key])
        train_rng.bit_generator.state = meta["train_rng"]
        replay_rng.bit_generator.state = meta["replay_rng"]
    return meta


# ---------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description="TavernAgent self-play")
    ap.add_argument("--steps", type=int, default=200, help="collect+train rounds")
    ap.add_argument("--games", type=int, default=128, help="self-play games per round")
    ap.add_argument("--sims", type=int, default=64, help="MCTS simulations per decision")
    ap.add_argument("--epochs", type=int, default=1, help="training passes per round")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-min", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="cap on training samples per step (split across workers)",
    )
    add_budget_args(ap)
    ap.add_argument("--horizon", choices=("turn", "game"), default="turn")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument(
        "--temperature-warm",
        type=float,
        default=1.0,
        help="temperature for the first --warm-steps rounds",
    )
    ap.add_argument("--warm-steps", type=int, default=5)
    ap.add_argument(
        "--policy-smoothing",
        type=float,
        default=0.05,
        help="fraction of the policy target spread over legal actions",
    )
    ap.add_argument(
        "--pg-weight",
        type=float,
        default=0.0,
        help="weight of the advantage-weighted policy gradient term",
    )
    ap.add_argument(
        "--noise",
        type=float,
        default=0.20,
        help="Dirichlet mixing at the search root during self-play",
    )
    ap.add_argument("--activation", choices=("gelu", "silu", "relu"), default="gelu")
    ap.add_argument("--residual-scale", type=float, default=1.0)
    ap.add_argument(
        "--head-norm", action="store_true", help="fixed LayerNorm before policy/value heads"
    )
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--noise-alpha", type=float, default=0.3)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument(
        "--value-samples",
        choices=("all", "end"),
        default="all",
        help="value supervision: all decisions (default) or only real selected EndTurn",
    )
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models" / "sp.npz"))
    ap.add_argument(
        "--probe-every",
        type=int,
        default=5,
        help="log the turn-1/turn-2 behaviour every N steps (0 = off)",
    )
    ap.add_argument("--probe-games", type=int, default=16)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--replay-rounds", type=int, default=1)
    ap.add_argument("--lr-decay-steps", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--persistent-workers", action="store_true")
    ap.add_argument("--save-trainer-state", action="store_true")
    ap.add_argument("--resume-state", default=None)
    args = ap.parse_args(argv)
    provided = {
        token.split("=", 1)[0][2:].replace("-", "_")
        for token in (sys.argv[1:] if argv is None else argv)
        if token.startswith("--")
    }
    if args.resume_state:
        with np.load(args.resume_state, allow_pickle=False) as snapshot:
            saved_meta = json.loads(str(snapshot["metadata"]))
        if "steps" not in provided or args.steps <= saved_meta["step"]:
            ap.error("--resume-state requires a target --steps beyond the saved step")
        for key, value in saved_meta["options"].items():
            if (
                key in vars(args)
                and key not in provided
                and key not in ("steps", "resume", "resume_state")
            ):
                setattr(args, key, value)

    if (
        min(
            args.steps,
            args.games,
            args.epochs,
            args.workers,
            args.max_samples,
            args.batch,
            args.sims,
        )
        < 1
    ):
        ap.error("steps/games/epochs/workers/max-samples/batch/sims must be positive")
    if not (0 <= args.noise <= 1 and 0 <= args.policy_smoothing <= 1):
        ap.error("noise and policy-smoothing must be in [0, 1]")
    if not (0 <= args.lr_min <= args.lr):
        ap.error("expected 0 <= lr-min <= lr")
    if args.replay_rounds < 1 or (args.lr_decay_steps is not None and args.lr_decay_steps < 1):
        ap.error("replay-rounds and lr-decay-steps must be positive")
    if args.resume and args.resume_state:
        ap.error("choose --resume (weights only) or --resume-state (full state)")
    if (args.save_trainer_state or args.resume_state) and args.replay_rounds != 1:
        ap.error("full-state checkpoints currently require replay-rounds=1")
    try:
        game_config(args.action_cap, args.move_cap, args.freeze_cap)
    except ValueError as exc:
        ap.error(str(exc))
    replay = ReplayWindow(args.replay_rounds)
    replay_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x5245504C]))
    args.workers = min(args.workers, args.games, args.max_samples)
    train_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x54524149]))
    net = (
        PolicyValueNet.load(args.resume)
        if args.resume
        else PolicyValueNet(
            seed=args.seed,
            activation=args.activation,
            residual_scale=args.residual_scale,
            head_norm=args.head_norm,
        )
    )
    if args.resume:
        for option in ("activation", "residual_scale", "head_norm"):
            actual = getattr(net, option)
            if option in provided and getattr(args, option) != actual:
                ap.error(f"--resume uses the checkpoint's {option}; conflicting option supplied")
            setattr(args, option, actual)
    trainer = Trainer(
        net,
        weight_decay=args.weight_decay,
        value_weight=args.value_weight,
        clip=args.clip,
        ema_decay=args.ema_decay,
        value_samples=args.value_samples,
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".work.npz")
    stats = Stats()
    history = []
    first_step = 1
    if args.resume_state:
        restored = load_training_state(args.resume_state, trainer, train_rng, replay_rng)
        first_step = restored["step"] + 1
        history = restored["history"]
        stats = Stats(**restored["stats"])
        args.save_trainer_state = True
        for option, attribute in (
            ("weight_decay", "wd"),
            ("clip", "clip"),
            ("value_weight", "value_weight"),
            ("ema_decay", "ema_decay"),
            ("value_samples", "value_samples"),
        ):
            if option in provided:
                setattr(trainer, attribute, getattr(args, option))
            else:
                setattr(args, option, getattr(trainer, attribute))
    else:
        net.save(str(out.with_name(out.stem + ".initial.npz")))

    out.with_suffix(".config.json").write_text(
        json.dumps(
            {**vars(args), **experiment_settings(args.action_cap, args.move_cap, args.freeze_cap)},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"net {net.n_params():,} params, {args.games} games x {args.steps} steps, "
        f"sims {args.sims}, workers {args.workers}",
        flush=True,
    )

    with ExitStack() as stack:
        persistent_pool = (
            stack.enter_context(mp.get_context("spawn").Pool(args.workers))
            if args.persistent_workers
            else None
        )
        for step in range(first_step, args.steps + 1):
            t0 = time.time()
            collect_cpu_start = time.process_time()
            net.save(str(tmp))
            seeds = [args.seed + step * 10000 + i for i in range(args.games)]
            chunks = [seeds[i :: args.workers] for i in range(args.workers)]
            per_worker = math.ceil(args.max_samples / args.workers)
            temperature = args.temperature_warm if step <= args.warm_steps else args.temperature
            jobs = [
                (
                    str(tmp),
                    chunk,
                    args.sims,
                    args.action_cap,
                    args.horizon,
                    temperature,
                    per_worker,
                    args.noise,
                    {
                        "noise_alpha": args.noise_alpha,
                        "c_puct": args.c_puct,
                        "move_cap": args.move_cap,
                        "freeze_cap": args.freeze_cap,
                    },
                )
                for chunk in chunks
            ]
            if persistent_pool is not None:
                results = persistent_pool.map(collect_chunk, jobs)
            else:
                with mp.get_context("spawn").Pool(args.workers) as pool:
                    results = pool.map(collect_chunk, jobs)
            samples, places = [], []
            rollout_cpu_seconds = 0.0
            rollout_decisions = 0
            for s, p, worker_metrics in results:
                samples.extend(s)
                places.extend(p)
                rollout_cpu_seconds += worker_metrics["cpu_seconds"]
                rollout_decisions += worker_metrics["decisions"]
            if len(samples) > args.max_samples:
                keep = train_rng.choice(len(samples), args.max_samples, replace=False)
                samples = [samples[i] for i in keep]
            fresh_count = len(samples)
            samples = replay.add_and_sample(samples, args.max_samples, replay_rng)
            gen = time.time() - t0
            collect_cpu_seconds = time.process_time() - collect_cpu_start + rollout_cpu_seconds

            t1 = time.time()
            train_cpu_start = time.process_time()
            pl = vl = 0.0
            for epoch in range(args.epochs):
                lr = cosine_lr(
                    args.lr,
                    args.lr_min,
                    (step - 1) * args.epochs + epoch,
                    (args.lr_decay_steps or args.steps) * args.epochs,
                )
                pl, vl = train_epoch(
                    trainer,
                    samples,
                    batch_size=args.batch,
                    lr=lr,
                    pg_weight=args.pg_weight,
                    smoothing=args.policy_smoothing,
                    rng=train_rng,
                )
            stats.policy_loss, stats.value_loss = pl, vl
            stats.games += len(places)
            stats.decisions += len(samples)
            train_seconds = time.time() - t1
            train_cpu_seconds = time.process_time() - train_cpu_start

            avg = math.ceil(len(samples) / args.batch) * args.epochs
            print(
                f"step {step:2d}: {len(samples):5d} samples from {len(places)} games "
                f"(gen {gen:5.1f}s, train {train_seconds:4.1f}s, {avg} batches)  "
                f"policy {pl:.3f}  value {vl:.3f}  lr {lr:.6g}",
                flush=True,
            )

            probe_start = time.time()
            probe_cpu_start = time.process_time()
            if args.probe_every and step % args.probe_every == 0:
                from .probe import behaviour_to_string

                print(
                    "  probe "
                    + behaviour_to_string(
                        net,
                        games=args.probe_games,
                        sims=args.sims,
                        horizon=args.horizon,
                        action_cap=args.action_cap,
                        move_cap=args.move_cap,
                        freeze_cap=args.freeze_cap,
                        c_puct=args.c_puct,
                    ),
                    flush=True,
                )

            history.append(
                {
                    **trainer.epoch_metrics,
                    "step": step,
                    "games": len(places),
                    "samples": len(samples),
                    "policy_loss": pl,
                    "value_loss": vl,
                    "lr": lr,
                    "fresh_samples": fresh_count,
                    "replay_rounds": len(replay.rounds),
                    "collect_seconds": gen,
                    "collect_cpu_seconds": collect_cpu_seconds,
                    "rollout_decisions": rollout_decisions,
                    "train_seconds": train_seconds,
                    "train_cpu_seconds": train_cpu_seconds,
                    "probe_seconds": time.time() - probe_start,
                    "probe_cpu_seconds": time.process_time() - probe_cpu_start,
                }
            )
            out.with_suffix(".history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            net.save(str(out))
            ema_net = PolicyValueNet(
                seed=0,
                embedding_dim=net.embedding_dim,
                activation=net.activation,
                residual_scale=net.residual_scale,
                head_norm=net.head_norm,
            )
            ema_net.p = {k: v.copy() for k, v in trainer.ema.items()}
            ema_net.save(str(out.with_name(out.stem + ".ema.npz")))
            if args.save_every and step % args.save_every == 0:
                net.save(str(out.with_name(f"{out.stem}.step{step}.npz")))
                ema_net.save(str(out.with_name(f"{out.stem}.step{step}.ema.npz")))
            if args.save_trainer_state:
                state_path = out.with_name(out.stem + ".trainer.npz")
                save_training_state(
                    state_path, trainer, step, train_rng, replay_rng, history, stats, args
                )
                if args.save_every and step % args.save_every == 0:
                    save_training_state(
                        out.with_name(f"{out.stem}.step{step}.trainer.npz"),
                        trainer,
                        step,
                        train_rng,
                        replay_rng,
                        history,
                        stats,
                        args,
                    )

    print(f"saved {out} and {out.with_name(out.stem + '.ema.npz')}", flush=True)
    (out.with_suffix(".stats.json")).write_text(
        json.dumps(
            {
                "games": stats.games,
                "decisions": stats.decisions,
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "params": net.n_params(),
                "sims": args.sims,
                "steps": args.steps,
                "games_per_step": args.games,
                "lr": args.lr,
                "wd": args.weight_decay,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
