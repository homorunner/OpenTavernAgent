# TavernAgent

TavernAgent is a policy/value network with PUCT search over TavernGym, trained by all-seat
self-play. It is the Rust crate `crates/tavernagent` and the `tavernagent` binary; it depends
on the `taverngym` crate directly and needs no Python. Checkpoints are numpy-readable `.npz`
files, so weights stay inspectable from Python.

The current model has **877,878 parameters**: a policy head and a value head. Weights are
model format v5; the search protocol is `ended-state-v3`.

## Commands

```bash
cargo build --release -p tavernagent
A=target/release/tavernagent

# Train 100 rounds (128 games; a third of the decisions searched at 256 simulations by default).
$A selfplay --steps 100 --workers 32 --save-trainer-state --save-every 50 --out runs/agent/model.npz

# Continue exactly from the saved trainer state to round 200; saved options apply unless
# given again on the command line.
$A selfplay --resume-state runs/agent/model.trainer.npz --steps 200

# Paired evaluation against seven scripted bots on the same seeds.
$A eval --model runs/agent/model.ema.npz --games 1024 --sims 64 --workers 32 --out runs/agent/eval.json

# Two checkpoints against each other (seat swaps, seed-clustered intervals); up to eight at
# one table; or a league of many checkpoints for rating fits.
$A arena --a runs/agent/model.npz --b runs/agent/model.step50.npz --games 512 --workers 32 --out runs/agent/arena.json
$A tournament --model a=runs/a.npz --model b=runs/b.npz --games 2048 --workers 32 --out runs/agent/tournament.json
$A league --model a_s100=... --model a_s200=... --rounds 2000 --workers 64 --out runs/league/league.jsonl
python tools/league_elo.py runs/league/league.jsonl --out images/elo.png   # Plackett-Luce ratings

# First actions of turns 1 and 2; per-turn action traces (Gold left at End Turn, loop
# candidates); inference and decision latency; a few games against the bots.
$A probe --model runs/agent/model.npz --games 32 --sims 64
$A patterns --model runs/agent/model.npz --games 64 --sims 64 --workers 32 --out runs/agent/patterns.json
$A bench
$A play --model runs/agent/model.npz --games 8 --sims 48
```

`--help` on any subcommand lists every option. `tavernagent eval/arena/tournament/league`
parallelise over games with rayon (`--workers`); `selfplay` additionally searches the
acting seats of each ply concurrently, so one slow game does not hold a round back.
Results do not depend on the thread count.

## Network and inputs

```
1713 numeric features + 69 card slots x 16-dim shared card embedding
  -> 192 -> 6 residual blocks  h <- h + GELU(LN(W h))
  -> 245 policy logits (masked softmax) / tanh value
```

- The environment exposes 3,235 floats and 124 card indices; the network drops the 1,519
  historical-opponent-warband floats, the 49 historical warband card slots and three
  redundant global flags (hero selection, game over, the Tavern-wide freeze flag), keeping
  the opponents' public summaries and heroes.
- The global block ends (before the lobby races) with `ended_turn`: the seat has finished
  recruiting and has no legal action. On such a position the value head estimates the
  warband that goes to combat.
- Minions, spells, heroes, hero powers, trinkets, quests and rewards share one embedding
  table indexed by `CardId + 1` (0 = empty; Dark Gifts are not embedded). The 69 slots are
  the own warband, hand, Tavern, the pending choice's source and options, trinkets,
  quests/rewards and the opponents' heroes; the Deity and secret slots are not used.
- Stats use the engine's `log1p` scaling; other features keep their linear scales.
  Host-side per-turn budgets (actions, Moves, Freeze presses) are not inputs: the search
  mask already excludes what is unaffordable, and the cap is a host safety limit, not a
  game rule.
- Hero selection is a uniform random choice outside the network (`heroes::select_heroes`,
  seeded); the chosen hero is still an input. Discovers, trinkets and targets are decided
  by the model.
- The backward pass is written by hand (`net.rs`); batched products use `matrixmultiply`
  (its AVX-512 kernels, one thread per call). `tests/net.rs` checks every gradient against
  central differences.
- **Search-time inference** (`PolicyValueNet::infer`) is separate from the training
  forward: the weight matrices are kept as a bfloat16 copy (`linalg::Bf16Mat`, half the
  bytes, so the whole network stays in a core's L2 cache), products accumulate in f32 in a
  hand-written AVX-512 kernel that skips zero inputs (runtime-dispatched, with a portable
  fallback), and GELU/SiLU use a clamped rational tanh. About 10 µs per evaluation against
  33 µs for the exact f32 path; the deviation from the f32 network is tiny (max policy
  probability difference ~4e-4, KL ~2e-7 on fresh weights). Training (`forward`/`backward`)
  stays exact f32; `infer_f32` is the bit-identical reference path. The bfloat16 copy is
  rebuilt lazily after `params_changed`, which `Trainer::step` calls.

`--hidden`, `--blocks`, `--embedding-dim`, `--activation gelu|silu|relu`,
`--residual-scale` and the parameter-free `--head-norm` define the architecture and are
stored in the checkpoint; `--resume` refuses conflicting options.

## Search

The default `--horizon turn` stays inside the seat's own recruit phase and never simulates
opponents or combat. End Turn is an ordinary edge: the simulation copy calls
`Game::end_turn_effects`, so end-of-turn triggers and their random choices happen and the
seat is marked ended without starting the fight; the resulting position is a leaf with no
legal actions, evaluated by the same value head. The search therefore compares "the
position after more actions" with "the position actually handed to combat", instead of
reusing a mid-turn value as the stopping value.

Each simulation uses an exact clone of the game with a fresh random stream; distinct
observable outcomes of one action (a Refresh, a Discover) become separate child nodes and
the action edge aggregates their visits. The real game and its random stream are never
touched.

The agent also plays under a must-spend rule (`mcts::MUST_SPEND_GOLD = 3`): a seat holding
at least 3 Gold may not end its turn, full stop. This is a training rule, not a game rule —
the engine still offers End Turn and the scripted bot still takes it. The check is a plain
Gold comparison rather than "is there something to buy", because the shop being empty right
now says nothing: a free Refresh may turn up something affordable, and that Refresh is
exactly what the seat should be pushed into. Since the rule keys on Gold alone it could empty
the mask, so one safety net remains: past the turn's action cap the engine returns End Turn
alone, and that single bit is restored. Gold does not carry over, so ending a turn with 3 or more
unspent Gold is always a wasted resource, and the early game converges faster when the policy
cannot learn to stop with a full purse.

- PUCT over all legal actions, unvisited actions at Q = 0. Default `--c-puct 1.9`; `--sims 64`
  in evaluation, 256 in self-play.
- Self-play searches a `--search-prob 0.33` fraction of the decisions and records only those
  (visit distribution as the policy target); the other decisions sample the network prior at
  the play temperature and leave no training row, like KataGo's full/cheap search split.
  Ended positions are recorded for every End Turn either way.
- The played action (evaluation, and self-play at temperature 0) is chosen by LCB after
  KataGo: among root actions with at least `--lcb-min-visit-prop 0.15` of the top visit
  count, the highest `Q - k * stdev(Q)` wins (`--lcb-stdevs 2`; 0 restores the plain visit
  argmax, ties: lowest index). Search protocol `ended-state-v4-lcb`.
- Self-play raises the root prior to `1 / --root-temperature` (1.1) and mixes Dirichlet
  noise into it (`--noise 0.20`, `--noise-alpha 0.3`); evaluation does neither.
- Per-turn host budgets (100 actions, 10 Moves, 5 Freeze presses) are enforced by the
  engine and reset every turn; `--action-cap/--move-cap/--freeze-cap` must be positive.
  The network does not see them.
- `--horizon game` advances the other seats with the scripted bot using hidden state; it is
  an experimental full-information mode only.
- One 64-simulation decision takes about 23 µs per simulation on one core, of which roughly
  10 µs is the network; the rest is the game copy, the action, the observation and the tree.

## Training

All eight seats play the current network. Every decision records the observation, legal
mask, root visit distribution, the action played and the final placement reward
`1 - 2 (placement - 1) / 7`. Each End Turn also records the ended position
(`end_turn_effects` on a copy): no policy target, the same value label. `--end-explore`
(0.02) forces End Turn with a small probability so that ended positions with unspent Gold
also appear; the search distribution stays the policy target.

**Loss.** Policy cross-entropy against the (smoothed, `--policy-smoothing 0.05`) visit
distribution, averaged over rows with legal actions; `--value-weight` (0.5) times the value
MSE over all rows. Optional `--pg-weight` adds a REINFORCE term with the value as baseline. AdamW (decay on
weight matrices only), global gradient-norm clipping, cosine learning-rate schedule.

**Parallel step.** A minibatch is processed in fixed chunks of `STEP_CHUNK_ROWS` (32) rows,
each running the forward and backward pass on its own thread (rayon); chunk gradients are
summed in chunk order and the gradient reduction and the AdamW/EMA update are split over
the parameter vector. The chunk size is a constant, so results do not depend on the number
of threads (`tests/training.rs` checks one against seven threads bit for bit). A 512-row
step takes about 12 ms on eight cores against 50 ms single-threaded; the elementwise code
(tanh, LayerNorm, assembling inputs, the optimizer) dominated, not the matrix products.

**Data.** Every decision of a round (about 200,000-240,000 from 128 games; the count grows
as the policy takes more actions per turn) enters a replay window of `--replay-rounds` (3)
rounds; nothing is discarded at collection. Each epoch (`--epochs 1`) trains on a uniform
draw of `--max-samples` (196,608) from the window, so a sample is used about once on
average over its three rounds in the window. The per-game cap that used to keep 65,536
decisions per round (about 28% of them) was removed.

**EMA.** An EMA copy of the weights is updated every minibatch with a decay derived from
`--ema-rounds` (2): `1 - 1 / (ema_rounds x updates per round)`, warmed up as
`min(decay, (1 + t) / (10 + t))`. It is saved as `<out>.ema.npz`.

| option | default |
| --- | --- |
| `--steps` / `--games` | 200 / 128 |
| `--sims` / `--c-puct` | 256 / 1.9 |
| `--replay-rounds` / `--epochs` / `--max-samples` / `--batch` | 3 / 1 / 65,536 / 512 |
| `--lr` / `--lr-min` / `--lr-decay-steps` | 3e-4 / 3e-5 / 200 |
| `--noise` / `--noise-alpha` / `--end-explore` | 0.20 / 0.3 / 0.02 |
| `--temperature-warm` (first `--warm-steps` 5) / `--temperature` | 1.0 / 1.0 |
| `--root-temperature` / `--lcb-stdevs` / `--lcb-min-visit-prop` | 1.1 / 2 / 0.15 |
| `--search-prob` | 0.33 |
| `--hidden` / `--blocks` / `--embedding-dim` | 192 / 6 / 16 |
| `--value-weight` | 0.5 |
| `--weight-decay` / `--clip` / `--ema-rounds` | 0.01 / 5.0 / 2 |

`--save-trainer-state` writes the network, AdamW moments, EMA, RNG states, history, options
and the **whole replay window** to `<out>.trainer.npz` after every round (about 6 GB at the
defaults, 9 KB per stored decision); `--save-every N` keeps immutable `<out>.stepN.npz` / `.ema.npz` / `.trainer.npz`.
`--resume-state` restores all of it bit-exactly (a smaller `--replay-rounds` keeps the newest
rounds); `--resume` loads weights only. `<out>.history.json` records per-round losses,
gradient norms, entropies, target KL, the ended sample counts and timings.

## Experiments so far

Settings: 128 games and 64 simulations per round, batch 512. Placement is the mean final
placement of seat 0 against seven scripted bots over 1,024 paired seeds (2,048 for the runs
below; the bot itself scores 4.60); Elo is a Plackett-Luce fit on eight-player leagues.

- **`c_puct` at 200 rounds is noise.** Two seeds ranked 1.3/1.5/1.7/1.9 in opposite orders.
  At 1,000 rounds (3e-4 with a 200-round cosine, then 3e-5) 1.9 led: placement 2.71 vs
  2.78 (1.5), 2.91 (1.3), 3.05 (1.7); in a 16,000-game league of the 100-round checkpoints
  1.9 was ahead from round 300 on (+146 Elo at 1,000 vs +124 for 1.5), hence the default.
- **Every run was still improving at 1,000 rounds**; all 1,000-round checkpoints beat all
  500-round ones.
- **The learning rate matters far more.** With a 1,000-round cosine, 6e-4 and 1.2e-3 fell
  below the bot for the first 400 rounds (placement 5.6 and 7.1 at round 200) and never
  caught up; 1.5e-4 was stable but slow. The best run so far spent 800 of its 1,000 rounds
  at 3e-5, which suggests the useful learning rate is small; lower peaks are being tested.
- **A 1,000-round cosine at 3e-4 is worse than annealing by round 200 and staying at 3e-5.**
  Both are about 1.8 placements better than the bot (2.71 and 2.81 against 4.60), but in a
  4,096-game paired arena the 200-round anneal wins by 24 Elo (network) and 18 Elo (EMA),
  95% CI [20, 29] and [13, 23] -- the placement difference between them is not significant
  on its own.
- **5,000 rounds, four seeds, 200-round anneal.** All four seeds still gained between 4,000
  and 5,000 rounds except the best one, which is flat from 4,000 on. Network placement at
  round 5,000 against the bots is 1.88 (seed 444444), 2.07 (333333), 2.11 (111111) and 2.13
  (222222). A 6,000-game league of the 20 EMA checkpoints (every 1,000 rounds, all four
  seeds, about 2,400 games each) gives, in Elo:

  | seed | 1,000 | 2,000 | 3,000 | 4,000 | 5,000 | EMA placement at 5,000 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 444444 | -35 | +28 | +54 | +90 | **+93** | 1.95 (EMA) |
  | 333333 | -141 | -34 | +8 | +44 | +70 | 2.08 |
  | 222222 | -95 | -15 | +15 | +40 | +48 | 2.19 |
  | 111111 | -96 | -81 | -40 | +5 | +44 | 2.10 |

  Two things follow. The seed spread at round 5,000 is about 49 Elo, wider than the gap
  between neighbouring checkpoints of one seed, so **one seed per configuration cannot rank
  configurations** -- which is what the earlier sections do. And a seed that is last at
  1,000 rounds ends third, so early rankings do not predict the final one.
- **Placement against the bots saturates near 2.0.** The four final checkpoints are 1.88 to
  2.13, inside two standard errors of each other (0.05 per checkpoint over 2,048 games);
  only the league separates them. A 2,048-game evaluation resolves about 0.1 placements
  (2 sigma); 1,024 games leave 0.06 per checkpoint and 0.11 on the difference of two.

## Open problems

- Ended-position value labels come only from where the policy actually stopped (plus
  `--end-explore`); asking "stop now" in mid-turn is still extrapolation. `patterns`'
  `mean_gold_left` at End Turn measures this directly — under the must-spend rule it cannot
  exceed 2 by construction, so high-Gold endings are now unreachable rather than measured.
- Policy, value and search gain are not evaluated separately; eight weak copies in
  self-play may give a thin learning signal early on.
- Dynamic card counters, attached cards and persistent effects are not fully encoded
  (`obs.rs` TODOs); opponents' historical warbands are not network inputs.
- The input projection is per slot (70% of the parameters): the same card in two Tavern
  slots uses different weights. A shared per-entity encoder would be the structural fix.

## Code map

| `crates/tavernagent/src/` | purpose |
| --- | --- |
| `net.rs`, `linalg.rs`, `npz.rs` | network and hand-written backward; matrices; numpy-compatible checkpoints |
| `mcts.rs` | PUCT, per-simulation random streams, chance branches, End Turn to the ended state |
| `selfplay.rs`, `train.rs` | self-play, AdamW/EMA, replay window, exact resume, the training loop |
| `eval.rs`, `arena.rs`, `tournament.rs`, `league.rs` | evaluation against the bot and between checkpoints |
| `probe.rs`, `patterns.rs`, `bench.rs`, `play.rs` | behaviour summaries, action traces, latency, games |
| `heroes.rs`, `rules.rs` | hero selection outside the network; host action budgets |

`cargo test --release -p tavernagent` covers gradients, input encoding, search boundaries,
ended samples, replay persistence and exact resume. `tools/league_elo.py`
fits and plots league ratings.
