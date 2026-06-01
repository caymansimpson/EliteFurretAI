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
- **Train over both regimes.** Already handled: `CurriculumConfig.open_team_sheets`
  defaults to `mixed`, which per-batch coin-flips the regime for both sides. No
  new work beyond confirming runs actually use `mixed` (the `0.5` flip
  probability is hardcoded; making it configurable is optional, not required).
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
- Land the OTS-robustness eval change first: dual-regime eval folded into
  `compute_score` (the implementation plan below). Mixed-regime training is
  already in place (`CurriculumConfig.open_team_sheets="mixed"`); just confirm
  the sweep configs use it. Without the eval change the later stages optimize
  against a regime the deployed agent will not always be in.
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

1. Implement dual-regime eval folded into `compute_score` (see the
   implementation plan below).
2. Confirm sweep training configs set `curriculum.open_team_sheets: mixed` so
   training exposes both regimes (already the default).
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
- 2026-05-31: Corrected an error in the OTS section. An earlier draft claimed
  training could not mix regimes without extending the flag; in fact
  `CurriculumConfig.open_team_sheets="mixed"` already per-batch coin-flips the
  regime and is the default. Only the eval side runs a single regime, so the
  remaining work is dual-regime eval. The wrong claim came from reading the
  2026-05-30 OTS design doc instead of the current code, which has moved past
  it. Added the implementation plan below for the eval change.

## Implementation Plan — Dual-Regime OTS Eval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let one eval pass score each opponent under both Open Team Sheets and closed-sheet play, and feed the worse (worst-case) win rate into `compute_score`, so a sweep cannot optimize one regime at the expense of the other.

**Architecture:** Add a third eval mode `"both"` to `EvalConfig.open_team_sheets`. When set, `_run_opponent_bucket` runs each opponent's full per-format cycle twice — once accepting OTS, once declining — and reports `win_rate = min(on, off)` as the scalar that feeds `compute_score`, while preserving both regimes' numbers in a new `BucketRunResult.per_regime` field for inspection and W&B logging. `"on"` / `"off"` keep today's single-regime behavior. vgc_bench is always forced ON inside `run_eval_parallel`, so under `"both"` it is run once and both regime entries are set equal (avoids wasting its battles). The score function, targets, and weights are unchanged — only the per-opponent win rate fed in changes.

**Tech Stack:** Python dataclasses, pytest with `unittest.mock` (the existing eval tests patch `run_eval_parallel` and `parse_player_specification`).

Activate the venv before every command: `source ../venv/bin/activate && <command>`.

---

### File Structure

Files modified (no new files):

- `src/elitefurretai/rl/config.py` — add `"both"` to `EVAL_OPEN_TEAM_SHEETS_MODES` (line 66) and update the `EvalConfig.open_team_sheets` docstring (lines 820-826). `from_dict` validation (lines 948-956) already checks membership in the tuple, so no logic change there.
- `src/elitefurretai/rl/analyze/evaluate_model.py` — add `per_regime` to `BucketRunResult` (lines 50-67); extract the per-format loop from `_run_opponent_bucket` into a `_run_one_regime` helper; make `_run_opponent_bucket` dispatch on the mode; add per-regime keys in `build_eval_log_payload` (lines 370-408) and `_serialize_result_to_dict` (lines 411+).
- `src/elitefurretai/rl/configs/example_eval.yaml` — flip `eval.open_team_sheets` to `"both"` with an explanatory comment.

Tests modified:
- `unit_tests/rl/analyze/test_evaluate_model.py` — extend `TestRunOpponentBucket`, `TestResultDataclasses`, and the payload tests.
- `unit_tests/rl/test_config.py` — eval mode validation.

---

### Task 1: Accept `"both"` as an eval OTS mode

**Files:**
- Modify: `src/elitefurretai/rl/config.py:66` and docstring at `src/elitefurretai/rl/config.py:820-826`
- Test: `unit_tests/rl/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `unit_tests/rl/test_config.py`:

```python
def test_eval_open_team_sheets_accepts_both():
    from elitefurretai.rl.config import EvalConfig

    cfg = EvalConfig.from_dict({"open_team_sheets": "both"})
    assert cfg.open_team_sheets == "both"


def test_eval_open_team_sheets_still_rejects_mixed():
    import pytest

    from elitefurretai.rl.config import EvalConfig

    with pytest.raises(ValueError, match="open_team_sheets"):
        EvalConfig.from_dict({"open_team_sheets": "mixed"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py::test_eval_open_team_sheets_accepts_both unit_tests/rl/test_config.py::test_eval_open_team_sheets_still_rejects_mixed -v`
Expected: `test_..._accepts_both` FAILS with `ValueError: eval.open_team_sheets must be one of ('off', 'on') ...`; the `rejects_mixed` test PASSES already.

- [ ] **Step 3: Add `"both"` to the eval mode tuple**

In `src/elitefurretai/rl/config.py:66` change:

```python
EVAL_OPEN_TEAM_SHEETS_MODES = ("off", "on")
```

to:

```python
EVAL_OPEN_TEAM_SHEETS_MODES = ("off", "on", "both")
```

Then update the `EvalConfig.open_team_sheets` docstring at lines 820-826 to describe `"both"`:

```python
    # Open Team Sheets for the eval pass: "on" (default), "off", or "both".
    # "both" runs every opponent under OTS and closed sheets and scores the
    # worse of the two (worst-case), so a checkpoint must hold up in both
    # regimes. "mixed" is still NOT allowed (that is a training-only blend).
    # vgc_bench is always evaluated ON regardless (its trained regime); under
    # "both" it is run once and both regime entries are set equal.
    open_team_sheets: str = "on"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_config.py::test_eval_open_team_sheets_accepts_both unit_tests/rl/test_config.py::test_eval_open_team_sheets_still_rejects_mixed -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/config.py unit_tests/rl/test_config.py
git commit -m "feat(eval): accept 'both' as an OTS eval mode"
```

---

### Task 2: Add `per_regime` to `BucketRunResult`

**Files:**
- Modify: `src/elitefurretai/rl/analyze/evaluate_model.py:50-67`
- Test: `unit_tests/rl/analyze/test_evaluate_model.py` (`TestResultDataclasses`)

- [ ] **Step 1: Write the failing test**

Add to `TestResultDataclasses` in `unit_tests/rl/analyze/test_evaluate_model.py`:

```python
    def test_bucket_run_result_per_regime_defaults_empty_and_accepts_values(self):
        empty = BucketRunResult(win_rate=0.5, n_battles=10)
        assert empty.per_regime == {}

        b = BucketRunResult(
            win_rate=0.55,
            n_battles=200,
            per_regime={"ots_on": 0.70, "ots_off": 0.55},
        )
        assert b.per_regime["ots_off"] == 0.55
        assert b.win_rate == min(b.per_regime.values())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source ../venv/bin/activate && pytest "unit_tests/rl/analyze/test_evaluate_model.py::TestResultDataclasses::test_bucket_run_result_per_regime_defaults_empty_and_accepts_values" -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'per_regime'`.

- [ ] **Step 3: Add the field**

In `src/elitefurretai/rl/analyze/evaluate_model.py`, add to the `BucketRunResult` dataclass (after the existing `per_format` field, around line 66):

```python
    # Format-weighted win rate per OTS regime ("ots_on" / "ots_off") when the
    # eval ran in "both" mode. Empty for single-regime ("on"/"off") passes.
    # The scalar `win_rate` above equals min(per_regime.values()) under "both".
    per_regime: Dict[str, float] = field(default_factory=dict)
```

(`field` and `Dict` are already imported in this file.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `source ../venv/bin/activate && pytest "unit_tests/rl/analyze/test_evaluate_model.py::TestResultDataclasses::test_bucket_run_result_per_regime_defaults_empty_and_accepts_values" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/analyze/evaluate_model.py unit_tests/rl/analyze/test_evaluate_model.py
git commit -m "feat(eval): add per_regime breakdown to BucketRunResult"
```

---

### Task 3: Extract a single-regime helper from `_run_opponent_bucket`

This is a pure refactor: behavior is unchanged, so the existing `TestRunOpponentBucket` tests must stay green.

**Files:**
- Modify: `src/elitefurretai/rl/analyze/evaluate_model.py:245-320`

- [ ] **Step 1: Add the helper above `_run_opponent_bucket`**

Insert this function immediately before `def _run_opponent_bucket(` (around line 245). It is the current per-format loop body with `accept_ots: bool` threaded into `run_eval_parallel` and the format-weighted win-rate math moved in:

```python
def _run_one_regime(
    opp_name: str,
    spec: OpponentEvalSpec,
    eval_cfg: EvalConfig,
    curriculum: CurriculumConfig,
    checkpoint_path: str,
    server_urls: List[str],
    device: str,
    run_tag: str,
    accept_ots: bool,
) -> Tuple[float, Dict[str, EvalResult], int]:
    """Run one opponent across all formats under a single OTS regime.

    Returns (format_weighted_win_rate, per_format, n_battles). vgc_bench is
    still forced ON inside run_eval_parallel regardless of accept_ots.
    """
    per_format: Dict[str, EvalResult] = {}
    format_weights: Dict[str, float] = dict(curriculum.battle_formats)
    battles_per_format = _split_battles_by_format(spec.n_battles, format_weights)

    for fmt in curriculum.battle_formats:
        n_for_fmt = battles_per_format.get(fmt, 0)
        if n_for_fmt <= 0:
            continue

        kwargs = _opponent_kwargs(opp_name, eval_cfg)
        if opp_name == "foul_play":
            kwargs["foul_play_team_pool_path"] = _foulplay_team_pool_for_fmt(
                eval_cfg, curriculum, fmt
            )

        model_spec = parse_player_specification(
            checkpoint_path, device=device, battle_format=fmt
        )
        opp_spec = parse_player_specification(
            opp_name, device=device, battle_format=fmt, **kwargs
        )

        cells = _build_cells_for_bucket(spec, opp_name, curriculum, fmt)
        per_format[fmt] = run_eval_parallel(
            p1=model_spec,
            p2=opp_spec,
            cells=cells,
            battles_per_cell=n_for_fmt,
            server_urls=server_urls,
            workers=eval_cfg.workers,
            run_tag=run_tag,
            collect_run_dir=eval_cfg.collect_trajectories,
            replay_sample_rate=eval_cfg.replay_sample_rate,
            open_team_sheets=accept_ots,
        )

    total_w = sum(format_weights[f] for f in per_format)
    if total_w > 0:
        win_rate = (
            sum(
                format_weights[f]
                * (per_format[f].player1_wins / max(per_format[f].battles_played, 1))
                for f in per_format
            )
            / total_w
        )
    else:
        win_rate = 0.0
    n_total = sum(r.battles_played for r in per_format.values())
    return win_rate, per_format, n_total
```

(`Tuple` is already imported in this file.)

- [ ] **Step 2: Rewrite `_run_opponent_bucket` to call the helper for the single-regime case**

Replace the body of `_run_opponent_bucket` (everything after the docstring, lines ~261-320) with:

```python
    t0 = time.time()
    accept_ots = eval_cfg.open_team_sheets == "on"
    regime_label = "ots_on" if accept_ots else "ots_off"
    win_rate, per_format, n_total = _run_one_regime(
        opp_name,
        spec,
        eval_cfg,
        curriculum,
        checkpoint_path,
        server_urls,
        device,
        run_tag,
        accept_ots=accept_ots,
    )
    return BucketRunResult(
        win_rate=win_rate,
        n_battles=n_total,
        per_format=per_format,
        per_regime={regime_label: win_rate},
        wall_time_s=time.time() - t0,
    )
```

(The `"both"` branch is added in Task 4. For now `"both"` would resolve `accept_ots` to `False`; that is fine because no test or config uses `"both"` yet.)

- [ ] **Step 3: Run the existing bucket tests to verify the refactor is green**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_evaluate_model.py::TestRunOpponentBucket -v`
Expected: all existing tests PASS (win-rate math and kwargs threading unchanged).

- [ ] **Step 4: Commit**

```bash
git add src/elitefurretai/rl/analyze/evaluate_model.py
git commit -m "refactor(eval): extract _run_one_regime from _run_opponent_bucket"
```

---

### Task 4: Run both regimes and score the worse one under `"both"`

**Files:**
- Modify: `src/elitefurretai/rl/analyze/evaluate_model.py` (`_run_opponent_bucket`)
- Test: `unit_tests/rl/analyze/test_evaluate_model.py` (`TestRunOpponentBucket`)

- [ ] **Step 1: Write the failing tests**

Add to `TestRunOpponentBucket` in `unit_tests/rl/analyze/test_evaluate_model.py`:

```python
    @patch("elitefurretai.rl.analyze.evaluate_model._resolve_opponent_team_text")
    @patch("elitefurretai.rl.analyze.evaluate_model._resolve_agent_team_text")
    @patch("elitefurretai.rl.analyze.evaluate_model.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.evaluate_model.parse_player_specification")
    def test_both_regimes_scores_min_and_records_per_regime(
        self, mock_parse, mock_run, mock_agent_team, mock_opp_team
    ):
        mock_parse.return_value = MagicMock()
        mock_agent_team.return_value = "AGENT_TEAM"
        mock_opp_team.return_value = "OPP_TEAM"
        # First regime pass (OTS on) wins 70/100; second (OTS off) wins 50/100.
        mock_run.side_effect = [
            EvalResult(
                label="ots_on", player1_wins=70, player2_wins=30, ties=0,
                battles_played=100,
            ),
            EvalResult(
                label="ots_off", player1_wins=50, player2_wins=50, ties=0,
                battles_played=100,
            ),
        ]
        eval_cfg = EvalConfig.from_dict({"open_team_sheets": "both"})
        spec = eval_cfg.opponents["max_damage"]
        result = _run_opponent_bucket(
            opp_name="max_damage",
            spec=spec,
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 1.0}),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        # run_eval_parallel called once per regime, with open_team_sheets True then False.
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0].kwargs["open_team_sheets"] is True
        assert mock_run.call_args_list[1].kwargs["open_team_sheets"] is False
        # Scalar win_rate is the worse regime (closed sheets here).
        assert result.win_rate == pytest.approx(0.50)
        assert result.per_regime["ots_on"] == pytest.approx(0.70)
        assert result.per_regime["ots_off"] == pytest.approx(0.50)

    @patch("elitefurretai.rl.analyze.evaluate_model._resolve_opponent_team_text")
    @patch("elitefurretai.rl.analyze.evaluate_model._resolve_agent_team_text")
    @patch("elitefurretai.rl.analyze.evaluate_model.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.evaluate_model.parse_player_specification")
    def test_both_mode_runs_vgc_bench_once(
        self, mock_parse, mock_run, mock_agent_team, mock_opp_team
    ):
        mock_parse.return_value = MagicMock()
        mock_agent_team.return_value = "AGENT_TEAM"
        mock_opp_team.return_value = ""
        mock_run.return_value = EvalResult(
            label="vgc_bench", player1_wins=60, player2_wins=40, ties=0,
            battles_played=100,
        )
        eval_cfg = EvalConfig.from_dict({"open_team_sheets": "both"})
        spec = eval_cfg.opponents["vgc_bench"]
        result = _run_opponent_bucket(
            opp_name="vgc_bench",
            spec=spec,
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 1.0}),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        # vgc_bench is forced ON internally, so it runs once, not twice.
        assert mock_run.call_count == 1
        assert result.win_rate == pytest.approx(0.60)
        assert result.per_regime["ots_on"] == pytest.approx(0.60)
        assert result.per_regime["ots_off"] == pytest.approx(0.60)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `source ../venv/bin/activate && pytest "unit_tests/rl/analyze/test_evaluate_model.py::TestRunOpponentBucket::test_both_regimes_scores_min_and_records_per_regime" "unit_tests/rl/analyze/test_evaluate_model.py::TestRunOpponentBucket::test_both_mode_runs_vgc_bench_once" -v`
Expected: both FAIL — `"both"` currently resolves to a single OTS-off pass, so `mock_run.call_count` is 1 in the first test (and the second regime assertion errors), and `per_regime` has only one key.

- [ ] **Step 3: Add the `"both"` branch to `_run_opponent_bucket`**

Replace the body written in Task 3 Step 2 with a dispatch on the mode:

```python
    t0 = time.time()

    if eval_cfg.open_team_sheets != "both":
        accept_ots = eval_cfg.open_team_sheets == "on"
        regime_label = "ots_on" if accept_ots else "ots_off"
        win_rate, per_format, n_total = _run_one_regime(
            opp_name, spec, eval_cfg, curriculum, checkpoint_path,
            server_urls, device, run_tag, accept_ots=accept_ots,
        )
        return BucketRunResult(
            win_rate=win_rate,
            n_battles=n_total,
            per_format=per_format,
            per_regime={regime_label: win_rate},
            wall_time_s=time.time() - t0,
        )

    # "both": vgc_bench is forced ON internally, so running it twice is wasted
    # work that yields identical numbers — run it once and mirror the regimes.
    if opp_name == "vgc_bench":
        win_rate, per_format, n_total = _run_one_regime(
            opp_name, spec, eval_cfg, curriculum, checkpoint_path,
            server_urls, device, run_tag, accept_ots=True,
        )
        return BucketRunResult(
            win_rate=win_rate,
            n_battles=n_total,
            per_format=per_format,
            per_regime={"ots_on": win_rate, "ots_off": win_rate},
            wall_time_s=time.time() - t0,
        )

    on_wr, on_fmt, on_n = _run_one_regime(
        opp_name, spec, eval_cfg, curriculum, checkpoint_path,
        server_urls, device, run_tag, accept_ots=True,
    )
    off_wr, off_fmt, off_n = _run_one_regime(
        opp_name, spec, eval_cfg, curriculum, checkpoint_path,
        server_urls, device, run_tag, accept_ots=False,
    )
    # The scalar that feeds compute_score is the worse regime. per_format is
    # taken from that same (scored) regime so the per-format breakdown matches
    # the number being scored; ties resolve to closed sheets.
    if off_wr <= on_wr:
        win_rate, per_format = off_wr, off_fmt
    else:
        win_rate, per_format = on_wr, on_fmt
    return BucketRunResult(
        win_rate=win_rate,
        n_battles=on_n + off_n,
        per_format=per_format,
        per_regime={"ots_on": on_wr, "ots_off": off_wr},
        wall_time_s=time.time() - t0,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_evaluate_model.py::TestRunOpponentBucket -v`
Expected: the two new tests PASS and all pre-existing bucket tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/analyze/evaluate_model.py unit_tests/rl/analyze/test_evaluate_model.py
git commit -m "feat(eval): score worst of both OTS regimes under 'both' mode"
```

---

### Task 5: Log per-regime win rates to W&B

**Files:**
- Modify: `src/elitefurretai/rl/analyze/evaluate_model.py` (`build_eval_log_payload`, lines 370-408)
- Test: `unit_tests/rl/analyze/test_evaluate_model.py`

- [ ] **Step 1: Write the failing test**

Add a new test class (or extend the existing payload test class) in `unit_tests/rl/analyze/test_evaluate_model.py`:

```python
class TestPayloadPerRegime:
    def test_payload_includes_per_regime_keys(self):
        ev = EvalResult(
            label="max_damage", player1_wins=50, player2_wins=50, ties=0,
            battles_played=100,
        )
        bucket = BucketRunResult(
            win_rate=0.50,
            n_battles=200,
            per_format={"gen9vgc2024regg": ev},
            per_regime={"ots_on": 0.70, "ots_off": 0.50},
        )
        result = MultiBucketEvalResult(
            per_bucket={"max_damage": bucket},
            score=-100.0,
            breakdown={"deficit_l2_pp": 100.0, "surplus_sum_pp": 0.0},
        )
        eval_cfg = EvalConfig()
        payload = build_eval_log_payload(result, update_step=5, eval_cfg=eval_cfg)
        assert payload["eval/max_damage/ots_on/win_rate"] == 0.70
        assert payload["eval/max_damage/ots_off/win_rate"] == 0.50
        assert payload["eval/max_damage/win_rate"] == 0.50
```

(Ensure `build_eval_log_payload` is imported in the test module's import block alongside the other `evaluate_model` symbols.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `source ../venv/bin/activate && pytest "unit_tests/rl/analyze/test_evaluate_model.py::TestPayloadPerRegime::test_payload_includes_per_regime_keys" -v`
Expected: FAIL with `KeyError: 'eval/max_damage/ots_on/win_rate'`.

- [ ] **Step 3: Emit the per-regime keys**

In `build_eval_log_payload`, in the loop over `result.per_bucket` (around lines 388-389), after the existing `payload[f"eval/{opp}/win_rate"] = bucket.win_rate` line, add:

```python
        for regime, regime_wr in bucket.per_regime.items():
            payload[f"eval/{opp}/{regime}/win_rate"] = regime_wr
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source ../venv/bin/activate && pytest "unit_tests/rl/analyze/test_evaluate_model.py::TestPayloadPerRegime::test_payload_includes_per_regime_keys" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/analyze/evaluate_model.py unit_tests/rl/analyze/test_evaluate_model.py
git commit -m "feat(eval): log per-OTS-regime win rates to wandb"
```

---

### Task 6: Persist per-regime numbers in the CLI JSON output

**Files:**
- Modify: `src/elitefurretai/rl/analyze/evaluate_model.py` (`_serialize_result_to_dict`, lines 411-440)
- Test: `unit_tests/rl/analyze/test_evaluate_model.py`

- [ ] **Step 1: Write the failing test**

Add to `unit_tests/rl/analyze/test_evaluate_model.py` (import `_serialize_result_to_dict`):

```python
class TestSerializePerRegime:
    def test_serialized_dict_includes_per_regime(self):
        ev = EvalResult(
            label="max_damage", player1_wins=50, player2_wins=50, ties=0,
            battles_played=100,
        )
        bucket = BucketRunResult(
            win_rate=0.50,
            n_battles=200,
            per_format={"gen9vgc2024regg": ev},
            per_regime={"ots_on": 0.70, "ots_off": 0.50},
        )
        result = MultiBucketEvalResult(per_bucket={"max_damage": bucket}, score=-100.0)
        out = _serialize_result_to_dict(
            result, checkpoint_path="/tmp/m.pt", config_path="/tmp/c.yaml", run_tag="abcd"
        )
        assert out["per_opponent"]["max_damage"]["per_regime"] == {
            "ots_on": 0.70,
            "ots_off": 0.50,
        }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source ../venv/bin/activate && pytest "unit_tests/rl/analyze/test_evaluate_model.py::TestSerializePerRegime::test_serialized_dict_includes_per_regime" -v`
Expected: FAIL with `KeyError: 'per_regime'`.

- [ ] **Step 3: Add `per_regime` to each opponent's serialized dict**

In `_serialize_result_to_dict` (lines 432-437), change the `per_opponent[opp]` assignment from:

```python
        per_opponent[opp] = {
            "win_rate": bucket.win_rate,
            "n_battles": bucket.n_battles,
            "wall_time_s": bucket.wall_time_s,
            "per_format": per_format_out,
        }
```

to:

```python
        per_opponent[opp] = {
            "win_rate": bucket.win_rate,
            "n_battles": bucket.n_battles,
            "wall_time_s": bucket.wall_time_s,
            "per_format": per_format_out,
            "per_regime": dict(bucket.per_regime),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `source ../venv/bin/activate && pytest "unit_tests/rl/analyze/test_evaluate_model.py::TestSerializePerRegime::test_serialized_dict_includes_per_regime" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elitefurretai/rl/analyze/evaluate_model.py unit_tests/rl/analyze/test_evaluate_model.py
git commit -m "feat(eval): include per-OTS-regime win rates in CLI json output"
```

---

### Task 7: Default the example eval config to `"both"`

**Files:**
- Modify: `src/elitefurretai/rl/configs/example_eval.yaml`

- [ ] **Step 1: Flip the flag with a comment**

In `src/elitefurretai/rl/configs/example_eval.yaml`, change the `open_team_sheets` line under `eval:` to:

```yaml
  # "both": score each opponent under OTS and closed sheets, feeding the worse
  # (worst-case) win rate into compute_score so the checkpoint must hold up in
  # both regimes. vgc_bench is always run ON and counted once.
  open_team_sheets: "both"
```

- [ ] **Step 2: Verify the config loads and round-trips**

Run: `source ../venv/bin/activate && python -c "from elitefurretai.rl.config import RNaDConfig; c = RNaDConfig.from_yaml('src/elitefurretai/rl/configs/example_eval.yaml'); print(c.eval.open_team_sheets)"`
Expected: prints `both` with no exception. (This is the exact load path the CLI uses — `evaluate_model.main` calls `RNaDConfig.from_yaml(args.config)` then reads `config.eval`.)

- [ ] **Step 3: Commit**

```bash
git add src/elitefurretai/rl/configs/example_eval.yaml
git commit -m "chore(eval): default example_eval to dual-regime OTS"
```

---

### Task 8: Full quality gate

- [ ] **Step 1: Run the eval + config test suites**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/analyze/test_evaluate_model.py unit_tests/rl/test_config.py -q`
Expected: all PASS.

- [ ] **Step 2: Lint and type-check**

Run: `source ../venv/bin/activate && ruff check src unit_tests && ruff format src unit_tests --check && pyright src/elitefurretai/rl/analyze/evaluate_model.py src/elitefurretai/rl/config.py`
Expected: no errors.

- [ ] **Step 3: Commit any formatting fixes**

```bash
git add -A
git commit -m "style(eval): lint/format fixes for dual-regime OTS eval"
```

---

### Out of scope / deferred

- Per-regime targets or weights (e.g. a laxer closed-sheet floor). The worst-case `min` keeps one target per opponent; revisit only if closed-sheet floors prove systematically harder.
- Making the training `mixed` flip probability (hardcoded `0.5`) configurable.
- Per-regime, per-format trajectory capture (`collect_trajectories`) — `"both"` writes both regimes' parquet under the same run dir today, which is acceptable for now.
