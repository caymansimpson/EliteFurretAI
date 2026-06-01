# RL Sweep Strategy — Staged Plan Toward Maximizing the Eval Score

## Context

Cayman is about to run a series of RL sweeps with the ultimate goal of
maximizing the W&B sweep metric defined in
[`compute_score`](../../src/elitefurretai/rl/analyze/evaluate_model.py#L27)
and configured in
[`example_eval.yaml`](../../src/elitefurretai/rl/configs/example_eval.yaml).
The target agent plays well against multiple scripted/learned heuristics,
across three formats (`gen9vgc2024regg`, `gen9vgc2026regi`,
`gen9championsvgc2026regma`), with multiple agent teams and against multiple
opponent teams.

A hard requirement: the agent must perform well in **both** Open Team Sheets
(OTS) and closed-sheet scenarios. OTS is therefore not a sweep parameter to
optimize over. It is a fixed dimension the worst-case score must hold across,
which shapes the eval harness, the training distribution, and the
representation work below. All three target formats are soft-OTS (per
[2026-05-30-09-04-ots-config-flag-design.md](2026-05-30-09-04-ots-config-flag-design.md)),
so both regimes are valid for each format.

Cayman's initial sweep plan had three axes:

1. **Algorithmic** — pure PPO vs RNaD vs Portfolio RNaD
2. **Training setup** — adaptive curriculum, types of teams
3. **Parameters** — `ent_coef`, `ppo_epochs`, `batch_size`, `lr`, etc.

This doc records the additional sweep axes worth considering and a staged,
fidelity-aware methodology for searching them, motivated by the shape of the
score function.

The sweep harness itself (W&B Bayesian search, Hyperband early termination,
dotted-key config patching, per-checkpoint multi-bucket eval) is designed in
[2026-05-29-21-00-rl-wandb-sweep-eval-design.md](2026-05-29-21-00-rl-wandb-sweep-eval-design.md)
and
[2026-05-29-21-30-rl-wandb-sweep-eval-implementation-plan.md](2026-05-29-21-30-rl-wandb-sweep-eval-implementation-plan.md).
This doc is about *what to sweep and in what order*, not the harness plumbing.

## Before State

- `compute_score` is hinge-L2 on per-opponent deficit plus linear surplus,
  weighted across opponents, format-averaged via `BucketRunResult.win_rate`.
  With `surplus_alpha=1.0`, any opponent or format below target dominates the
  scalar.
- Per-opponent targets in `example_eval.yaml` are stricter than graduation:
  80% vs SHP, 80% vs MaxDamage, 80% vs BCPlayer, 60% vs VGCBench. FoulPlay is
  weight 0.0 (broken).
- Hyperparameters are tuned by hand today; the Bayesian sweep harness is being
  built but the search space has not been mapped.
- The model
  ([`TransformerThreeHeadedModel`](../../src/elitefurretai/supervised/model_archs.py#L876))
  is a causal transformer in the ps-ppo / decision-transformer style. Each
  turn's encoded state is a token in a sequence with sinusoidal positional
  encoding and a causal mask (`use_causal_mask=True`, `max_seq_len=40`). At RL
  inference, [rl_trajectory_player](../../src/elitefurretai/rl/rl_trajectory_player.py#L99)
  accumulates a per-battle context tensor across turns (`forward_with_hidden`,
  `hidden_states` keyed by `battle_tag`, reset at battle end). The model
  therefore **already captures cross-turn information** — it attends causally
  over the turn history. There are two attention mechanisms: the
  `GroupedEncoder`'s `pokemon_cross_attn` over the six Pokémon *within* a
  state, and the main `TransformerEncoder` over turns *across* time.
- `embedder_feature_set` defaults to `"raw"`; the embedder also supports
  `SIMPLE` and `FULL` ([embedder.py](../../src/elitefurretai/etl/embedder.py#L179)).
- `open_team_sheets` is a first-class config flag with different mode sets per
  context. **Training already supports mixing both regimes**:
  `CurriculumConfig.open_team_sheets` accepts `off` / `on` / `mixed`, defaults
  to `mixed`, and `mixed` does a per-batch coin flip applied to both sides and
  shared opponents (config.py `open_team_sheets_for_battle`, applied in
  opponents.py around line 1141). vgc_bench is always forced ON. So the
  training-distribution half of the OTS requirement is done.
- **Eval runs a single regime.** `EvalConfig.open_team_sheets` allows only
  `off` / `on` (no `mixed`), and `evaluate_model._run_opponent_bucket` passes
  one boolean into `run_eval_parallel`. There is no way to score both regimes
  in one eval pass or fold both into `compute_score`. This is the remaining
  work for the OTS-robustness requirement.
- Opponent types are enumerated in
  [`OpponentType`](../../src/elitefurretai/rl/opponents.py#L145): `SELF_PLAY`,
  `BC_PLAYER`, `EXPLOITERS`, `GHOSTS`, plus the scripted/external baselines.

## Problem

The score is a **worst-case / robustness objective**, not an average-
performance objective. The deficit term is quadratic and the surplus term is
linear, so:

- A single sub-target opponent or format costs more than the gain from
  overperforming everywhere else combined.
- Seed variance feeds the squared deficit term, so a noisy run is scored as if
  it were a bad config.

A three-axis one-factor-at-a-time sweep over (algorithm, training setup,
scalar params) misses the axes that most directly move worst-case performance,
and it ignores the cost asymmetry between structural choices (expensive to
sweep) and continuous hyperparameters (cheap to joint-tune). Without a staged
plan, the sweep budget gets spent ranking noise and re-discovering
interactions.

## Solution

### Additional sweep axes, ranked by leverage for this objective

**A. Representation / observation.** Usually higher leverage than the RL
algorithm. Concrete knobs:
- `embedder_feature_set` (`raw` / `full` / `simple`) and feature ablations.
- Whether the policy consumes belief/inference features (posterior over hidden
  items, moves, spreads from the inference module) versus raw observables. In
  an imperfect-information game this is often the single largest win, and the
  OTS-robustness requirement raises its priority: closed-sheet play is exactly
  where the agent must *infer* hidden team info, while OTS hands it over. A
  representation that expresses "known vs inferred" consistently — known fields
  filled directly under OTS, filled from the posterior otherwise — is the
  cleanest way to make one model strong in both regimes. Note that OTS itself
  is a fixed dual requirement here, not a knob to sweep (see the OTS subsection
  below).

**B. Architecture.** Distinct from both the RL algorithm and scalar params.
Cross-turn memory already exists (the causal transformer over turn-states), so
the architecture work is tuning that sequence model rather than adding
recurrence:
- Context window. `max_seq_len=40` caps how many turns the model can attend
  over; VGC battles can run longer. Sweep the window length and confirm long
  battles are not silently truncated.
- Transformer capacity: `transformer_layers`, `transformer_heads`,
  `transformer_ff_dim`, and the causal-mask / positional-encoding scheme.
- Within-state entity attention in the `GroupedEncoder`
  (`pokemon_attention_heads`, `grouped_encoder_*` dims).
- Trunk sharing (shared vs separate actor/critic; `sep_arch.yaml` exists),
  layer widths/depths (`early_layers`, `late_layers`, head layer stacks).
- Model size under a fixed compute budget (a small scaling sweep).

**C. Opponent / self-play population.** This axis *is* the robustness
objective made concrete during training.
- Self-play vs league/PFSP weighting toward opponents currently being lost to.
- Mix ratio of scripted baselines (`max_damage`, `simple_heuristic_baseline`,
  `bc_player`) inside the training distribution vs holding them out for eval.
- Frozen-checkpoint pool (`GHOSTS`) composition and sampling weights;
  `EXPLOITERS` count (`max_exploiter_models`) and refresh cadence.
- Because the deficit term is quadratic per opponent, "train more against the
  matchup you're losing" maps almost one-to-one onto the metric. Portfolio
  RNaD is well-motivated for the same reason: it is a robustness method applied
  to a robustness objective.

**D. Reward shaping and horizon.** Win/loss is sparse over long VGC episodes.
- Intermediate shaping (HP-differential, KO/faint bonuses, win-margin vs
  binary terminal reward).
- `gamma` and `gae_lambda`.
- Decoupled from the eval metric, so it can be tuned aggressively without
  contaminating the objective.

**E. Warm-start and anchoring.**
- BC init from `rose-sun-108-mega_best.pt` vs scratch, and the strength /
  annealing of any KL-to-BC anchor.
- RNaD-specific: regularization coefficient, anchor update cadence, NeuRD vs
  standard PG. These interact strongly with `ent_coef` and must be jointly
  tuned, not swept independently.

**F. Variance control and checkpoint selection.** Not training knobs, but they
gate the trustworthiness of every result above.
- Seed replication. Self-play RL is high-variance and the deficit term is
  squared, so a single run can mis-rank a config. Finalists need at least 3
  seeds before their score is trusted.
- Checkpoint selection policy: last vs best-on-proxy vs weight-averaged
  (EMA/SWA). Averaging tends to buy robustness for free, which is what the
  deficit term rewards.

### OTS robustness is a fixed dual requirement, not a sweep axis

Because the agent must be strong in both OTS and closed-sheet play, OTS is
folded into the objective rather than searched. This adds work that must land
*before* the sweeps start, because a sweep run against a single regime will
trade away the other:

- **Eval over both regimes.** `compute_score` should see each opponent in both
  OTS and closed-sheet, and the deficit term should take the worse of the two
  (or include both as separate buckets so the floor must hold in each). This
  roughly doubles eval cost and makes OTS another axis of the worst-case min,
  alongside opponent and format.
- **Train over both regimes.** Training only OTS-on lets the policy lean on
  revealed team info and collapse when sheets are closed (and vice versa). The
  training distribution must mix both. Today's flag is one global value per
  run, so this needs the flag extended to a per-battle / probabilistic regime
  with pool coordination (soft OTS drops the battle unless both sides agree).
- **Represent the regime consistently.** The cleanest path is the belief-
  feature design in axis A: identical feature slots filled from revealed sheets
  under OTS and from the inference posterior otherwise, so the model sees one
  observation schema and the regime is just a confidence difference. This
  avoids training two effectively different policies behind one set of weights.

### Staged, fidelity-aware methodology

The axes interact and differ by cost, so search them in stages rather than as
one flat sweep. Earlier stages use cheap proxy metrics and short runs; the full
multi-opponent `compute_score` eval is reserved for finalists.

**Stage 0 — prerequisites, proxy metric, and seed-noise baseline.**
- Land the OTS-robustness harness changes first: dual-regime eval folded into
  `compute_score`, and a mixed-regime training distribution. Without these the
  later stages optimize against a regime the deployed agent will not always be
  in.
- Pick a cheap proxy that correlates with `compute_score`: win rate vs one fast
  scripted baseline (e.g. `max_damage` or `simple_heuristic_baseline`) at a
  fixed, short step budget, or sample efficiency to a fixed win-rate. Measure
  the proxy in both OTS regimes so it does not hide a one-regime collapse.
- Run the current best config across 3-5 seeds to measure the score's
  seed-noise floor. Any sweep delta smaller than this band is not real.

**Stage 1 — structural choices (coarse grids, proxy metric).** Representation
(A) and architecture (B), plus the algorithm family from Cayman's axis 1. These
are expensive to change and define the ceiling, so fix them first with coarse
grids judged on the proxy. The belief-feature / known-vs-inferred
representation is the highest-priority item here, since it is also what makes
one model robust across both OTS regimes. Carry forward the few configurations
that are clearly on the frontier rather than a single winner.

**Stage 2 — opponent distribution and reward (medium fidelity).** Population /
self-play (C) and reward shaping (D), conditioned on the Stage 1 frontier.
Evaluate on a reduced multi-opponent proxy (all four opponents but fewer
battles per opponent) since these axes directly move worst-case matchups.

**Stage 3 — continuous hyperparameters (Bayesian + Hyperband).** Joint Bayesian
sweep over `lr` (and `backbone_lr` / `heads_lr`), `batch_size`, `ent_coef` /
`ent_coef_end`, `ppo_epochs`, `clip_range`, `vf_coef`, `max_grad_norm`,
`gae_lambda`, plus RNaD regularization and anchor cadence from (E). Joint, not
one-at-a-time, because LR↔batch and entropy↔RNaD-reg optima move together. Use
Hyperband/ASHA to kill weak trials early.

**Stage 4 — finalist confirmation (full eval, multi-seed).** Re-run the top
handful from Stage 3 on the full `compute_score` eval (full battle counts, all
three formats) across ≥3 seeds. Rank by the floor, then add EMA/SWA checkpoint
selection (F) on the winner.

### Selection rule across all stages

Rank by the floor, not the mean. The scalar already encodes this, but the
per-opponent, per-format, and per-OTS-regime breakdown in
`MultiBucketEvalResult.per_bucket` must be inspected: a config with higher mean
win rate but a newly regressed sub-target matchup — including a regime that
quietly collapsed in OTS or closed-sheet — scores worse and should be rejected.
A higher headline number that hides a dropped floor is not an improvement.

## Reasoning

- The score's quadratic-deficit / linear-surplus shape is the central fact.
  Every prioritization above (population matching, regularization for
  stability, ensembling/Portfolio RNaD, EMA checkpoints, seed replication)
  derives from it being a robustness objective.
- Representation and architecture sit above the algorithm in leverage because a
  weak observation caps performance no matter how the policy gradient is
  computed. Imperfect information and turn history are the defining features of
  VGC. The causal transformer already models turn history, so the open
  architecture questions are whether its context window and capacity are
  sufficient, not whether memory exists at all. Observation quality (belief
  features, OTS) is the larger remaining lever.
- Staging by cost avoids spending the expensive full eval on structural
  exploration, and joint-tuning the continuous params avoids the interaction
  traps that defeat one-factor-at-a-time search.
- The proxy-metric and seed-noise baseline in Stage 0 exist so that later
  stages can distinguish real deltas from variance, which the squared deficit
  term would otherwise amplify into false rankings.
- OTS is a fixed dual requirement rather than a sweep axis because the agent
  must hold up in both regimes. Folding both into the score and the training
  mix before sweeping prevents the search from optimizing a regime the deployed
  agent will not always be in. It does not reorder the stages, but it adds a
  Stage 0 prerequisite and promotes belief features to the top of Stage 1.

## Planned Next Steps

1. Extend the eval harness to score both OTS regimes per opponent and fold the
   worse (or both as buckets) into `compute_score`.
2. Extend the OTS flag from one global value to a per-battle / probabilistic
   regime with pool coordination, so training mixes OTS and closed-sheet.
3. Decide the Stage 0 proxy metric (measured in both regimes) and run the 3-5
   seed noise baseline on the current best config.
4. Map each axis above to concrete dotted-key entries in the RL sweep YAML
   schema so the harness can patch them.
5. Define per-stage W&B sweep configs (search method, early-termination
   bracket, parameter ranges) reusing the supervised `train_sweep.py` patterns.
6. Decide which opponents/formats stay in the reduced Stage 2 proxy eval vs the
   full Stage 4 eval.

## Updates

- 2026-05-31: Initial version. No experiments run yet; this records the sweep
  axis taxonomy and staged methodology agreed in discussion.
- 2026-05-31: Corrected the architecture section. An earlier draft claimed the
  model had no cross-turn memory; it is in fact a causal transformer over
  per-turn state tokens with a 40-turn context accumulated across turns at
  inference. Reframed architecture axis B from "add recurrence" to "tune the
  existing sequence model" (context window, capacity), and dropped the
  now-moot verification step. The original claim came from grepping config for
  LSTM/GRU instead of reading the model forward.
- 2026-05-31: Added the OTS-robustness constraint. The agent must perform well
  in both OTS and closed-sheet play, so OTS is a fixed dual requirement folded
  into the score and training mix rather than a sweep axis. Added a dedicated
  subsection, a Stage 0 prerequisite (dual-regime eval + mixed-regime
  training), promoted belief features to the top of Stage 1, and added the two
  harness next-steps that must land before sweeping.
