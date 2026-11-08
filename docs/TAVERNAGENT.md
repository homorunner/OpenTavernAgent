# TavernAgent

TavernAgent is a policy/value network with PUCT search over TavernGym, trained by all-seat
self-play. It is the Rust crate `crates/tavernagent` and the `tavernagent` binary; it depends
on the `taverngym` crate directly and needs no Python. Checkpoints are numpy-readable `.npz`
files, so weights stay inspectable from Python.

The default model has **2,315,637 parameters**, with a linear policy head and a nonlinear
value head. Model format is v11, trainer format is 9, and the search protocol is
`ended-state-v9-compact-play`.

## Commands

```bash
cargo build --release -p tavernagent
A=target/release/tavernagent

# Train 100 rounds (128 roots per round, binary split before turn-4 combat;
# a third of decisions searched at 256 simulations; 2 epochs, constant lr 3e-5).
$A selfplay --steps 100 --workers 32 --save-trainer-state --save-every 50 --out runs/agent/model.npz

# Continue exactly from the saved trainer state to round 200; saved options apply unless
# given again on the command line.
$A selfplay --resume-state runs/agent/model.trainer.npz --steps 200

# Paired evaluation against seven scripted bots on the same seeds.
$A eval --model runs/agent/model.ema.npz --games 1024 --sims 64 --workers 32 --out runs/agent/eval.json

# Same evaluation against the fixed, table-free teacher policy.
$A eval --model runs/agent/model.ema.npz --opponent teacher --games 2048 --sims 64 --workers 32 --out runs/agent/teacher_eval.json

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
1708 numeric features + 16 turn one-hot bits + 57 card slots x 32-dim card embedding + 116 mask bits
  -> Linear(3664, 256) -> LayerNorm -> GELU
  -> 6 pre-norm residual blocks:
       h <- h + scale * (W_down GELU(W_up LayerNorm(h) + b_up) + b_down)
       256 -> 384 -> 256
  -> policy: Linear(256, 116) -> masked softmax
  -> value: Linear(256, 128) -> GELU -> Linear(128, 1) -> tanh
```

- The policy head has **116 actions**: indices 0–45 match the engine; indices 46–115
  encode `Play(hand, pos)` as `46 + hand * 7 + pos`. A compatible occupied position
  resolves to the engine's Magnetize action (including on a full board); otherwise it
  resolves to normal Play. Appending at the empty right edge plays a Magnetic minion
  separately. Spells still use position 0. Move is absent.
  `actions::engine_action` translates against the current state, and the mask uses the
  same translation. TavernGym's 245-action API is unchanged.
- The FFN width is fixed to `hidden + ceil(hidden / 2)`. `W_up` uses He initialization;
  `W_down` uses He divided by `sqrt(2 * blocks)` to bound the initial residual variance
  as depth increases.
  The residual scale remains 1, head norm remains off, and the policy head remains linear.
  The value hidden width is fixed at 128 and uses GELU.
- The raw turn scalar (`turn / 20`) is replaced inside TavernAgent by 16 one-hot bits:
  T1 through T15 have separate bins; T16 and later share the last bin. The engine
  observation schema is unchanged; pre-game turn 0 maps to T1.
- The 116-bit `agent_mask` enters the shared trunk as 0/1 features, so both policy and
  value can use legality. Training and inference use the same mask; ended states use
  an all-zero mask. The ended-turn flag remains in the original numeric input.
- The environment exposes 3,235 floats and 124 card indices; the network drops the 1,519
  historical-opponent-warband floats, the 49 historical warband card slots and three
  redundant global flags (hero selection, game over, the Tavern-wide freeze flag), keeping
  the opponents' public summaries and heroes.
- The global block ends (before the lobby races) with `ended_turn`: the seat has finished
  recruiting and has no legal action. On such a position the value head estimates the
  warband that goes to combat.
- Minions, spells, heroes, hero powers, trinkets, quests and rewards share one embedding
  table indexed by `CardId + 1` (0 = empty; Dark Gifts are not embedded). The 57 slots are
  the own warband, hand, Tavern, the pending choice's source and options, trinkets,
  quests/rewards and the opponents' heroes; the Deity and secret slots are not used.
- Quest inputs retain only the first active quest, its target reward and the first owned
  reward (three card slots), plus the first quest's presence and progress. Additional
  quests/rewards, including their reward-generated trinkets, are omitted from network input.
- Only two ordinary owned-trinket slots enter the network. Trinkets acquired through a
  hero power or quest reward occupy that power's or owned reward's card slot instead.
  Resolved Orb of the Unknown shells are skipped; their generated trinkets inherit the
  acquisition source. Pending Orb options remain visible. Acquisition sources are recorded
  per trinket instance; replacement and Orb generation preserve them. Engine effects
  stay unchanged.
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
- **Search-time inference** uses a thread-local reusable workspace. After warmup,
  internal inference allocates nothing and `predict` allocates only its returned policy.
  Policy logits and the value hidden projection share one matrix product; output bias,
  scaling and residual addition are fused. GELU/SiLU use a vectorized rational tanh
  with AVX-512 runtime dispatch.
- The input projection skips zero inputs and uses BF16 weights with FP32 activations
  and FMA accumulation. Dense FFN layers use native AVX-512 BF16 dot products where
  available, rounding activations to BF16 and accumulating in FP32. Unsupported CPUs
  retain the BF16-weight/FMA path (AVX-512, AVX2, or portable). The head projection uses
  BF16-weight/FMA; the final 128-to-1 value projection uses an FP32 dot product.
  Packed dot weights are built lazily and invalidated by `params_changed` along with
  the other inference weights. Training stays FP32; `infer_f32` is the exact reference.
- BF16 reduces weight traffic, but does **not** guarantee L2 residency. The experiment
  CPU has 1 MiB L2 per physical core; the default model's BF16 matrix weights alone
  exceed it. Active input rows, shared L3, model depth and concurrency all affect speed.
  Native BF16 also introduces activation rounding; verify value error, policy KL and
  action margins on trained checkpoints, not just initialized weights.
- Reproduce the FMA/native-BF16 and fresh/reused-workspace comparison with
  `cargo run --release -p tavernagent --example inference -- --blocks 6,8,10,12`.
  Add `--workers 128 --iters 5000` for throughput, or `--model PATH` for trained weights.
  It reports JSON lines on bot-driven T1/T4/T8/T12 observations and valid ended previews;
  it measures network inference, not complete search or self-play throughput.


On experiment (EPYC 9976), a 128-thread, 71-observation preflight measured **2.42M →
2.62M predictions/s** for six blocks and **1.16M → 1.40M/s** for twelve blocks when
switching the reusable-workspace FFN kernel from FMA to native BF16. These are short
network-only benchmarks. The allocation test proves zero intermediate allocations;
fresh/reused timing differences were small or noisy, so no allocator speedup is claimed.
On a 12-block checkpoint after eight small training rounds, native BF16's maximum
value error versus FP32 was **0.00244**, mean policy KL **2.09e-7**, with no policy argmax
flips across those observations. Long-trained checkpoint precision still needs monitoring.

`--hidden`, `--blocks`, `--embedding-dim`, `--activation gelu|silu|relu`,
`--residual-scale` and the parameter-free `--head-norm` define the architecture and are
stored in the checkpoint; `--resume` refuses conflicting options.

## Search

Search stays inside the seat's own recruit phase and never simulates
opponents or combat. End Turn is an ordinary edge: the simulation copy calls
`Game::end_turn_effects`, so end-of-turn triggers and their random choices happen and the
seat is marked ended without starting the fight; the resulting position is a leaf with no
legal actions, evaluated by the same value head. The search therefore compares "the
position after more actions" with "the position actually handed to combat", instead of
reusing a mid-turn value as the stopping value.

Each simulation starts from the same game state with a fresh random stream. Search reuses a working copy and rewinds recruit-state changes between simulations,
falling back to a full clone when necessary. Distinct observable outcomes of one action
(a Refresh, a Discover) become separate child nodes, and the action edge aggregates their
visits. The real game and its random stream are never touched.

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

A second training rule in the same `agent_mask`: the agent may not Cancel. A target menu
(a hand spell, a Hero Power, an Activate) or a Choose One the agent opened must be resolved by
picking an option. The engine refuses to open a target menu with no target, so an option is
always available; Cancel comes back only when it is the sole legal action (a paid trinket
offer with nothing affordable). Without it the search treats open-menu / close-menu as a free
no-op pair — `patterns` showed up to 49 consecutive `Play`/`Cancel` pairs in one turn,
burning the action budget for nothing. The rule removes the agent's option to back out of a
choice; the plugin's human player keeps it.

Move is not a rule but an absence: it is outside the agent's action space altogether (see
"Network and inputs").

- PUCT over all legal actions. An unvisited action is valued at the parent's mean minus
  `--fpu-reduction` (default 1.5; negative restores the old fixed Q = 0). Default
  `--c-puct 1.9`; `eval` defaults to 48 simulations and self-play to 256. The recorded
  primary evaluations explicitly used `--sims 64 --c-puct 1.5`.
- Self-play searches a `--search-prob 0.33` fraction of the decisions and records only those
  (visit distribution as the policy target); the other decisions sample the network prior at
  the play temperature and leave no training row, like KataGo's full/cheap search split.
  Ended positions are recorded for every End Turn either way.
- The played action (evaluation, and self-play at temperature 0) defaults to the most
  visited root action (`--lcb-stdevs 0`, ties: lowest index). Optional LCB selection uses
  the highest `Q - k * stdev(Q)` among actions with at least `--lcb-min-visit-prop 0.15`
  of the top visit count. Positive-temperature self-play samples the visit distribution
  instead, so LCB does not affect training at the default temperature of 0.5.
- Self-play raises the root prior to `1 / --root-temperature` (1.1) and mixes Dirichlet
  noise into it (`--noise 0.20`, `--noise-alpha 0.3`); evaluation does neither.
- Per-turn host budgets (100 actions, 10 Moves, 5 Freeze presses) are enforced by the
  engine and reset every turn; `--action-cap/--move-cap/--freeze-cap` must be positive.
  The network does not see them.
- Use `bench` to measure inference and search latency on the target machine; costs depend
  on the checkpoint, search budget, CPU instruction set and position.

## Training

All eight seats play the current network. Every searched decision records the observation, legal
mask, root visit distribution, the action played and the final placement reward
`1 - 2 (placement - 1) / 7`. Each End Turn also records the ended position
(`end_turn_effects` on a copy): no policy target, the same value label.

`--end-explore` (0.25) selects a fraction of root games for one forced End Turn.
Uniformly choose one of the eight seats and an action index from **1 through 32**.
Count all of that seat's decisions, including cheap actions and pending choices;
at the selected index, force End Turn. If it is illegal (including the must-spend-Gold
mask), defer until the next legal End Turn. A normal End Turn also counts and consumes
the intervention if selected. No planning rollout or replay is used.

Binary splitting remains enabled: exactly one randomly chosen child inherits a pending
intervention and its counter; after it fires, neither child can fire again. Discard the
selected seat's earlier rows on the intervention path, including shared ancestors. Keep
the forced decision, ended-position row, later rows, all other seats' rows and unaffected
siblings. If the seat dies or the game ends before the intervention can fire, retain all
rows; the actual fraction with a forced end can therefore be below 25%.

The intervention gets a search even if originally cheap, so its policy target remains
the search distribution, not a one-hot End Turn target. Other seats' value labels still
include their opponent's intervention; they are not unbiased for intervention-free play.

**Tree rollouts.** Self-play defaults to `--split-turn 4`: each root splits once into
two continuations after turn 4. `selfplay --split-turn "4,8"` splits each root into two
continuations after turn 4 and each surviving continuation into two after turn 8. A split happens
after all seats' end-of-turn effects and recruit eliminations, immediately before combat:
that turn's combat is already independent in the children. Both environment and policy/search
random streams are reseeded deterministically per child. The shared prefix is stored once;
its value labels average the two child subtree returns with probability 1/2 each. Thus,
when all branches reach both splits, turns 1–4 average four outcomes, turns 5–8 average
two, and later rows use their own terminal outcome. A branch that ends early returns its
outcome directly; tree probabilities, rather than a flat mean of unequal leaf counts,
keep the label expectation unchanged. Ended-position rows of the split turn belong to
the shared prefix.

Every split multiplies **both policy and value sample weights** by the code constant
`selfplay::SPLIT_SAMPLE_WEIGHT = 0.55` (no CLI option). For `4,8`, weights are 1, 0.55,
and 0.3025 across the three stages. The multiplier affects training only, never the
probabilities used to average labels; it gives each additional depth 1.1 times the
aggregate sample weight when both children survive. `--games` still counts independent
roots, and reported placement averages subtree returns per root. Replay draws remain
uniform over stored rows; loss weighting applies after the draw. Collection cost and
replay size grow with the continuations, while `--max-samples` still bounds each epoch's
training draw. Splits execute depth first within each root, retaining parallel seat
searches and parallel roots. Pass `--split-turn` without values for ordinary rollouts;
turns must be positive, unique and increasing (for example `--split-turn "3,6,9"`).

**Loss.** Policy cross-entropy against the (smoothed, `--policy-smoothing 0.05`) visit
distribution, normalized by the sum of weights of rows with legal actions; `--value-weight`
(0.5) times value MSE normalized by the sum of all row weights. Optional `--pg-weight` adds
a REINFORCE term with the same policy row weights and the value as baseline. AdamW (decay on
weight matrices only), global gradient-norm clipping, and a constant learning rate by
default (`--lr 3e-5 --lr-decay-steps 0`). Set a positive `--lr-decay-steps` and a lower
`--lr-min` to enable cosine decay; `--lr-min` is unused when decay is disabled.

**Parallel step.** A minibatch is processed in fixed chunks of `STEP_CHUNK_ROWS` (32) rows,
each running the forward and backward pass on its own thread (rayon); chunk gradients are
summed in chunk order and the gradient reduction and the AdamW/EMA update are split over
the parameter vector. The chunk size is a constant, so results do not depend on the number
of threads (`tests/training.rs` checks one against seven threads bit for bit). Compare
`train_seconds` and optimizer updates alongside placement when changing batch size.

**Data.** Every recorded search decision and ended position enters a replay window of
`min(6, 3 + step / 200)` rounds (integer division using the cumulative training round):
3 initially, 4 at round 200, 5 at 400, and 6 from 600. `--replay-rounds N` overrides
this with a fixed window. Each epoch (`--epochs 2`) trains on a uniform draw of
`--max-samples` (80,000) from the window. Tree rollouts add rows without increasing this
training cap, so they also change the average reuse of a row at the same epoch count.

**EMA.** An EMA copy of the weights is updated every minibatch with a decay derived from
`--ema-rounds` (2): `1 - 1 / (ema_rounds x updates per round)`, warmed up as
`min(decay, (1 + t) / (10 + t))`. It is saved as `<out>.ema.npz`.

| option | default |
| --- | --- |
| `--steps` / `--games` | 200 / 128 |
| `--sims` / `--c-puct` | 256 / 1.9 |
| `--replay-rounds` / `--epochs` / `--max-samples` / `--batch` | scheduled 3→6 / 2 / 80,000 / 512 |
| `--lr` / `--lr-min` / `--lr-decay-steps` | 3e-5 / 3e-5 (unused without decay) / 0 (constant) |
| `--noise` / `--noise-alpha` / `--end-explore` | 0.20 / 0.3 / 0.25 |
| `--temperature-warm` (first `--warm-steps` 5) / `--temperature` | 0.5 / 0.5 |
| `--root-temperature` / `--lcb-stdevs` / `--lcb-min-visit-prop` | 1.1 / 0 / 0.15 |
| `--search-prob` / `--fpu-reduction` | 0.33 / 1.5 |
| `--split-turn` | 4 (one binary split) |
| `--hidden` / `--blocks` / `--embedding-dim` | 256 / 6 / 32 |
| `--value-weight` | 0.5 |
| `--weight-decay` / `--clip` / `--ema-rounds` | 0.01 / 5.0 / 2 |

`--save-trainer-state` writes the network, AdamW moments, EMA, RNG states, history, options
and the **whole replay window** to `<out>.trainer.npz` every **10 cumulative rounds** by
default. `--save-trainer-every N` changes that interval (positive; use 1 for every round).
The final round always saves trainer state when enabled. `--save-every N` additionally
saves immutable `<out>.stepN.npz` / `.ema.npz` / `.trainer.npz` at archive rounds, including
when they fall between trainer intervals; trainer archives require trainer state saving.
Raw and EMA weight files still update every round. Snapshot size depends on the number of
stored rows, including tree continuations. Array payloads use bounded 64 KiB writes with
the same little-endian NPY encoding.
`--resume-state` restores all of it bit-exactly (a smaller `--replay-rounds` keeps the newest
rounds) and enables trainer state saving; the default window schedule continues from the
saved round. It inherits the saved trainer interval unless explicitly overridden; older
snapshots without an interval use 10. An interrupted run resumes at its last saved round,
so work since that snapshot must be repeated.
`--resume` loads weights only. `<out>.history.json` records per-round losses,
gradient norms, entropies, target KL, the ended sample counts and timings. Losses and
per-epoch metrics currently describe the last training epoch, not an independent
validation set; `samples` counts that epoch's draw, not the total across all epochs.
Trainer snapshots include the network, replay row weights and saved options; loading
validates the model/search schema and replay completeness. Saved split turns are inherited
on resume unless explicitly overridden. The multiplier is recorded as
`split_sample_weight` in `<out>.config.json`; history includes
`sample_weight_sum` and `policy_sample_weight_sum` alongside the raw sample counts.

## Experiment results (2026-11-06–08)

These runs used the model structures and frozen binaries described in each cohort.
They predate the current turn one-hot encoding, nonlinear value head and compact
quest/trinket inputs; their results do not evaluate the current architecture.

After the FPU correction, both cohorts used 128 games/round, 256 search simulations,
batch 512, max-samples 65,536, seed 444444444, constant learning rates and no rollout
splitting. E1 used one epoch, replay 3 and FPU=1.0; e2 used two epochs, replay 5 and
FPU=1.5, so their differences cannot be attributed to epochs alone. EMA placement against
seven scripted bots over 2,048 shared seeds (64 simulations, c-puct=1.5, LCB=2 and each
run's FPU setting; lower is better):

| Run / learning rate | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e1 / 3e-5 | 2.177 | 2.013 | 1.688 | 1.799 | 1.601 | — |
| e1 / 1e-4 | 1.948 | 1.743 | 1.762 | 1.670 | 1.588 | 1.550 |
| e2 / 3e-5 | 1.913 | 1.652 | 1.590 | — | — | — |
| e2 / 1e-4 | 1.833 | 2.111 | 2.230 | — | — | — |

E1 stopped at round 2970 for lr=3e-5 and 3046 for lr=1e-4. Dashes indicate no completed
evaluation; the last evaluated checkpoints were rounds 2500 and 3000 respectively.

- **E1 kept improving after round 1500.** The 3e-5 run recovered from a dip at round 2000;
  the 1e-4 run also improved, so the higher rate did not cause general regression.
  At the same round 2500 their difference was only −0.013 placements (1e-4 minus 3e-5;
  paired 95% CI [−0.075, 0.048]). The 1e-4 gain from 2500 to 3000 was also inconclusive
  (−0.038, CI [−0.099, 0.024]); later progress was slower. Late replay value MSE stayed
  near 0.25 in both runs despite improving placement.
- **E2 was more sensitive to learning rate.** In 512-game arenas, the 3e-5 round-1500
  model scored 64.3% against round 500, but only 52.1% against round 1000. At 1e-4,
  round 1500 scored 47.6% against round 500. Scores compare cross-model placements,
  not first-place rates.
- **LCB hurt e2 evaluation.** On 1,024 paired seeds at 64 simulations, disabling LCB
  improved round-1500 placement from 1.607 to 1.538 (3e-5), and 2.227 to 2.023 (1e-4).
  Temperature-1 self-play does not use LCB action selection.
- **E2 training loss was misleading at 1e-4.** From round 500 to 1500, replay value MSE fell
  from 0.226 to 0.206 while MSE on fresh self-play rose from 0.246 to 0.268. Increasing
  batch size should be tested alongside training intensity, not treated as a proven fix.

### Tree rollout / batch / epochs comparison (2026-11-06–07)

Four runs on `experiment`, from commit `3eb0302` and identical initial weights:
A = batch 512 / epochs 1, B = 1024 / 1, C = 512 / 2, D = 1024 / 2.
All completed 3,000 rounds with 128 roots/round, split-turn=4 (child weight 0.55),
256 simulations, 128 workers, replay 5, max-samples 65,536, constant lr=3e-5,
temperature=1, FPU=1.5, LCB=0 and seed 444444444. Remote raw artifacts were
removed in the 2026-11-07 cleanup.

**Data utilization decreased.** Early collection produces roughly 140k rows/round
versus 78k without splitting, but each epoch still draws only 65,536 rows from replay.
At fixed epochs, expected reuse per collected row is therefore about 44% lower in
steady state; extra generated rows do not imply proportionally more training updates.
E2 collection takes about 19 s/round versus 8 s previously, while gradient training
remains about 1.7 s. These historical timings also reflect different learned policies;
this is not a comparison at equal compute cost. All four runs kept 128 roots/round.
Across the long runs, collection accounts for 93–98% of collection-plus-training time.
E1 draws about 0.44 training rows per fresh row and E2 about 0.88; these are average
reuse counts, not independent-sample counts. More epochs are relatively cheap here.

Every 500 rounds: raw/EMA evaluation over 2,048 games and fresh value validation on
64 independent, unsplit games. Pairwise EMA arenas use 512 games at rounds 500, 1,500
and 3,000. EMA placements (2,048 games, one training seed) are:

| Run | Round 500 | Round 1,000 | Round 1,500 | Round 2,000 | Round 2,500 | Round 3,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: batch 512 / epochs 1 | 1.895 | 1.674 | 1.522 | 1.548 | 1.568 | 1.471 |
| B: batch 1024 / epochs 1 | 2.048 | 1.753 | 1.578 | 1.399 | 1.378 | 1.400 |
| C: batch 512 / epochs 2 | 1.695 | 1.646 | 1.615 | 1.468 | 1.371 | 1.368 |
| D: batch 1024 / epochs 2 | 1.810 | 1.727 | 1.658 | 1.498 | 1.434 | 1.409 |

All four improved beyond round 1,500. At 3,000, direct play orders **C > D > A > B**,
as at 2,500; a separate 2,048-game C-versus-D confirmation scores **52.3% (51.3–53.2%)**.
Against their own round-2,500 checkpoints, the four round-3,000 models score
**50.3% / 52.3% / 52.8% / 51.1%** in 2,048-game arenas. A is inconclusive
(49.3–51.2%); the other three intervals exceed 50%. C's nearly unchanged bot placement
therefore hides continued progress. B's bot placement also overstates its strength
relative to A/D. Arena scores compare cross-team placements, not first-place rates;
these intervals condition on one training seed per recipe.

A/B/C/D make **128/64/256/128 optimizer updates per round**. Increasing batch alone
halves updates; A and D match updates while D draws twice as many rows. C's total
collection-plus-training time is 18.73 h versus A's 17.88 h (about 5% more), including
differences in trajectories and machine contention. The results favor testing data
reuse and learning rate jointly before increasing batch. They do not isolate splitting's
variance benefit from its extra data, or validate the new temperature-.5/lr-1e-4 recipe.

In 128-game bot diagnostics, C's turn-2 upgrade frequency rises from 2.3% at round 500
to 40.6% at 3,000; mean turn-10 Tavern tier rises from 2.73 to 4.07. Strategy continues
changing, but local behavior is still suspect: C3000 ends turn 1 with an empty board in
72/128 games, including 66 where it buys without playing any card (50 are simply
`Buy → EndTurn`). These are traced actions, not proof of an engine bug. A useful next
diagnostic is to compare EndTurn/Play estimates with repeated continuations from fixed
early positions, and track empty-board/buy-without-play rates in the new recipes.
Fresh unsplit value MSE remains roughly .23–.26 and early-turn R² is low;
neither establishes an irreducible noise floor. Replay loss is not directly comparable
because split-prefix labels average continuations and training uses sample weights.
All 90 scheduled evaluations completed successfully, alongside the additional arenas
and diagnostics.

### Borrowed-VM parameter campaign (2026-11-06–07)

Finished at **Nov 7 10:55 CST**, ahead of the noon deadline. The final audit reached
all **310 borrowed hosts** and verified no experiment directories or related processes
remained. One timed-out FPU replicate was excluded from complete-group analyses.

A 100-setting C_PUCT / temperature / noise grid used three training seeds per setting;
important candidates received five- or ten-seed follow-ups. The table below uses ten
paired seeds, 400 rounds × 128 roots, and 64 training workers. Its reference is the
pre-campaign CLI baseline: 256 simulations, epochs 1, lr=3e-5, temperature 1 and replay 3. Arena scores
compare cross-team placements; 50% means parity, not first-place win rate. Intervals
use training seeds as replicates and are descriptive, without selection/multiplicity
correction. Cheap-search refinements reused the ten seeds and also had five fresh seeds.
Standard arenas use 64 simulations, C_PUCT=1.5, FPU=1.5 and LCB=0.

**Starting presets.** The balanced recipe is now the CLI default for new runs;
resumed runs inherit their saved settings unless explicitly overridden.

| Preset | Simulations / epochs / learning rate | Arena vs old CLI (95% interval) | Time / old CLI |
| --- | --- | ---: | ---: |
| Balanced: next general-training baseline | 256 / 2 / 1e-4 | **77.7%** (74.6–80.8%) | **1.10×** |
| Higher endpoint quality | 256 / 8 / 3e-5 | **78.9%** (76.2–81.6%) | **1.28×** |
| Fast parameter experiments | 64 / 4 / 3e-5 | **70.9%** (67.7–74.0%) | **0.39×** |

All three use temperature **0.5**, including `--temperature-warm 0.5`, batch **512**,
replay **5**, split-turn **4**, LCB **0**, constant learning rate and max-samples **65,536**.
The 256-simulation presets use C_PUCT=**1.9**, noise=**.2**, FPU=**1.5**; the fast preset
uses C_PUCT=**2.5**, noise=**.1**, FPU=**.5**. Keep value-weight=.5, policy-smoothing=.05,
root-temperature=1.1 and noise-alpha=.3. Time ratios measure collection plus optimizer
work, excluding evaluation/serialization; VM and learned-trajectory differences remain.

E8 beats the balanced E2 recipe **51.5%** (50.1–52.9%) at round 400, for **17%** more
time. Its small advantage also appears at rounds 100/200/300 and with 256-simulation
evaluation. The fast E4 recipe beats cheap E2 **56.2%** (54.8–57.6%) for 20% more time.
Against the balanced recipe's round-200 model it scores **50.6%** (46.8–54.5%) using
78% of the time; this does not establish equivalence. Against its round-400 model it
scores only **41.1%**, so cheaper search is a quality/cost tradeoff.

Temperature .5 is the clearest exploration change: ten-seed isolated confirmation
scores **61.7%** against temperature 1; five-seed C/noise controls show no further
reliable gain. Continuation from one shared C500 initialization also favors lower
temperature. Matched-update batch-1024 tests show no reliable advantage over 512;
learning-rate decay likewise has no demonstrated benefit in the continuation tests.

Search budget, epochs and loss weights must be tuned together. Cheap-search E8 was
unstable; 128-simulation E8 regressed badly in both ten- and five-seed cohorts while
E1/E2/E4 remained healthy. At full search/E8, raising value-weight from .5 to 1 scored
only **17.7%** against the control across three seeds. These models had finite weights;
lower training loss alone does not establish stronger play. Split changes data quantity,
weights and reuse as well as target variance. Further temperature reduction (.25/.1)
and other loss/prior changes remain candidates, not confirmed defaults. The subsequent
campaign compared network depth with E2 and constant lr=3e-5.

### Pre-norm FFN depth comparison (2026-11-07)

Trained four v10 models from scratch on TavernGym 37.0.2, comparing **6/8/10/12
blocks** at width256, embedding32 and FFN width384. All used 128 roots, sims256,
search probability .33, splitT4, E2, lr3e-5, batch512, max-samples80k and replay3→6.
Each run had 128 physical cores and one training seed.

At the common **800-round** checkpoint, EMA evaluation with 64 search simulations
produced the following mean placements (lower is better). Teacher is the AutoTavern
rule policy without composition tables.

| Blocks | Parameters | Teacher (95% interval) | Scripted | Collection + training hours |
| --- | ---: | ---: | ---: | ---: |
| 6 | 2.378M | **1.687 ± .050** | **1.274** | **6.87** |
| 8 | 2.774M | 1.867 ± .056 | 1.373 | 8.67 |
| 10 | 3.169M | 1.845 ± .055 | 1.354 | 9.79 |
| 12 | 3.565M | 1.757 ± .050 | 1.353 | 11.66 |

Teacher results use 4,096 games per model. Timing excludes checkpoint I/O and
evaluation. The runs stopped at **1181 / 993 / 971 / 866 rounds**. The 6-block model
continued improving: at round1100 it scored approximately **1.523 against teacher**
and **1.215 against scripted**.

Six blocks gave the best observed quality and speed; deeper models cost 26–70% more
collection/training time at round800 without better placement. This single-seed
comparison does not establish that depth itself hurts: initialization and self-play
trajectories differed. Gradient and BF16 checks found no obvious numerical failure.
This motivated the following continuation experiment from one shared starting policy.

### Identity-depth continuation (2026-11-08)

Compared 6/8/10/12-block models starting from the same **6-block step1181 EMA**,
with added blocks initialized as identity mappings. Optimizer and replay were reset;
all runs used lr=3e-5, E2, batch512, max-samples80k and splitT4. Training stopped after
roughly 300–400 additional rounds.

EMA mean placement (lower is better), evaluated with 64 search simulations:

| Model | Teacher +100 | Teacher +200 | Teacher +300 | Scripted +300 |
| --- | ---: | ---: | ---: | ---: |
| Common starting policy | 1.503 | — | — | 1.196 |
| 6 blocks | 1.551 | 1.518 | 1.492 | 1.186 |
| 8 blocks | 1.582 | 1.474 | 1.526 | 1.196 |
| 10 blocks | 1.521 | 1.517 | 1.509 | 1.189 |
| 12 blocks | 1.583 | 1.576 | 1.533 | 1.193 |

Teacher +300 used 4,608 games per model (2,560 for 12 blocks); paired-seed 95%
intervals showed no significant improvement over the starting policy. The 6-block
+400 result was also flat at **1.497**. Arena showed a temporary advantage for 8-block
+200 that weakened at +300. With one training seed, the current conclusion is a
plateau under this recipe, with no sustained benefit from adding depth.

The main current suspicion is **data quality caused by v10's `end-explore=0.02`**:
every eligible decision could force End Turn, and the resulting placements were also
used to label earlier states. Repeated interventions may distort the value targets
and prevent further improvement; this explanation still needs a controlled experiment.

## Open problems

- Validate value calibration on early-stop states: `--end-explore` samples one stop
  within the first 32 actions (or the next legal end), so coverage remains limited.
- Search improvement under visit-count selection and the fixed FPU reduction needs
  controlled comparisons. Tree splits also change sample counts, weights and data reuse.
- **TODO: value auxiliary targets.** Test short-term combat/damage targets alongside the
  nonlinear value head. Keep final expected placement as search utility.
- **TODO: input representation.** Test a shared per-card/entity encoder that retains
  zone and position information. Current input projection remains per slot; simply
  pooling away warband order would discard strategically relevant information.

Idle evaluation slots add independent 512-game seed batches to the latest checkpoints,
up to 16,384 eval games per checkpoint **per opponent** (scripted and teacher independently). Spare slots run 128-game league batches until
each model has at least 16,384 seat appearances. New checkpoint evaluations take priority
at job boundaries. Reports pool all games; league uncertainty resamples whole games. Repeated monitoring uses descriptive 95% intervals,
not a sequential significance test. Cards show only progress, placement intervals and plots.

League selection uses `0.2/N + 0.8 * (count+256)^-1 / sum((count+256)^-1)`
for each model, sampling without replacement and shuffling seats. Counts include completed
games and exact reserved lineups of running/queued batches, and increase after each planned
table. Failed jobs release their reservations. The league executable accepts these explicit
tables; the training/scripted-eval executable and inference code remain unchanged.
