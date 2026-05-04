# Context

This note records a targeted `100`-battle embedder profile on the Showdown websocket backend after the safe CPU batching winner had already been identified.

The purpose was to validate which embedder-focused optimization ideas are actually worth doing next, using the current recommended Stage 2 CPU actor settings rather than short synthetic tests.

# Before State

- the previous benchmark and profile work showed that embedding plus damage calculation was still a first-order bottleneck in the Showdown actor loop
- we had several plausible ideas for speeding up `feature_set=full`
- the main open question was which of those ideas would produce meaningful wall-clock gains versus looking good only in theory

# Problem

We needed to rank embedder optimization directions using real `100`-battle evidence, not intuition.

In particular, we needed to know whether to focus next on:

1. exact damage feature reduction or caching
2. stat-bound caching
3. move-availability shortcuts
4. broader structural embedder caching

# Solution

Added a dedicated profiling wrapper around the existing Showdown benchmark path and ran:

- `policy=model`
- `config=src/elitefurretai/rl/configs/single_team_showdown_profile_longer_no_vgcbench.yaml`
- `battles=100`
- `device=cpu`
- `batch_size=8`
- `batch_timeout=0.02`
- `max_concurrent_battles=4`

Observed benchmark summary:

- `duration_seconds=168.157`
- `battle_loop_seconds=162.190`
- `battles_per_second=0.595`

Observed embedder timing summary from `/tmp/showdown_embedder_profile_report.json`:

- `Embedder.embed_to_array = 35.770s` over `2813` calls
- `Embedder.embed = 31.912s` over `2816` calls
- `Embedder.generate_feature_engineered_features = 22.087s` over `2822` calls
- `embedder.calculate_damage = 17.664s` over `79,610` calls
- `Embedder.generate_opponent_pokemon_features = 4.777s`
- `Embedder.generate_pokemon_features = 3.675s`
- `embedder.compute_stats = 2.013s` over `113,366` calls
- `Pokemon.available_moves_from_request = 0.076s` over `7,546` calls

Supporting `cProfile` evidence:

- `players._embed_battle_state` cumulative time was about `36.048s`
- `embedder.calculate_damage` cumulative time was about `20.813s`
- `generate_feature_engineered_features` cumulative time was about `19.131s`

# Reasoning

This profile makes the next optimization priorities much clearer.

The dominant embedder cost is not general feature assembly. It is the exact damage machinery inside the engineered feature block.

Practical ranking:

1. **Highest priority: reduce or cache exact damage work**
   - direct timing evidence shows this is the largest isolated embedder hotspot
   - because `calculate_damage` sits inside `generate_feature_engineered_features`, improvements here should cut both the damage subtotal and much of the engineered-feature total
   - likely highest-payoff directions are request-local memoization, skipping non-damaging moves, and replacing non-critical bench interactions with coarse threat buckets

2. **Second priority: broader engineered-feature simplification / active-board-only exactness**
   - the engineered feature block totals `22.087s`, and only part of that is the raw damage call itself
   - this means there is still meaningful overhead in surrounding loops, temporary team insertion, KO calculations, and repeated cross-product feature assembly
   - active-board exact features plus cheaper bench summaries should recover a large fraction of this cost while preserving most VGC-relevant signal

3. **Third priority: cache stat bounds by species**
   - `compute_stats` is real work and is called very often, but at only about `2.013s` over the full run it is much smaller than damage work
   - still worth doing because it is low risk and easy to cache by `(species, level, min|max)`
   - expected payoff is modest on its own, but it stacks well with damage optimization

4. **Low priority: optimize move availability checks**
   - `available_moves_from_request` consumed only about `0.076s`
   - this is too small to be a serious focus area for throughput work
   - it can still be cleaned up later for code simplicity, but not as a performance priority

Estimated wall-clock implications:

- a strong damage-focused pass is the only embedder change likely to recover on the order of `10s+` over `100` battles
- stat-bound caching alone is more likely in the low single-digit seconds range per `100` battles
- move-availability optimization is effectively negligible for end-to-end throughput

# Planned Next Steps/Implementation Plan

1. Implement request-local damage memoization keyed by attacker, defender, move, boosts, tera, weather, field, and major side-condition state.
2. Add a config-controlled actor runtime mode that keeps exact damage for current active-board interactions and replaces most bench cross-product damage features with coarse threat buckets.
3. Cache opponent min/max stat bounds by species and level so `compute_stats` is not recomputed in the hot path.
4. Re-run the same `100`-battle benchmark after each of the first two changes and compare `battle_loop_seconds`, `Embedder.generate_feature_engineered_features`, and `embedder.calculate_damage`.

# Follow-Up Experiment: Request-Local Damage Memoization

Ran a concrete before/after test of the first proposed optimization using the same `100`-battle Showdown benchmark shape.

## Baseline Variance (`3` runs, no code change)

Command shape:

- `src/elitefurretai/engine/analyze/showdown_benchmark.py`
- `policy=model`
- `config=single_team_showdown_profile_longer_no_vgcbench.yaml`
- `battles=100`
- `device=cpu`
- `batch_size=8`
- `batch_timeout=0.02`
- `max_concurrent_battles=4`

Observed baseline variance:

- `duration_seconds`: `94.701`, `95.881`, `101.840` (mean `97.474`)
- `battle_loop_seconds`: `88.820`, `90.060`, `96.041` (mean `91.640`)

Interpretation:

- the `100`-battle benchmark still has material noise, but the baseline band is tight enough that a `10s+` shift in battle-loop time is clearly real

## After Implementing Conservative Damage Cache (`3` runs)

The tested cache used conservative battle-state keys and a bounded cache size.

Observed after-change variance:

- `duration_seconds`: `102.809`, `110.483`, `111.207` (mean `108.166`)
- `battle_loop_seconds`: `96.839`, `104.631`, `105.349` (mean `102.273`)

Relative to baseline mean:

- `duration_seconds` got worse by about `10.692s`
- `battle_loop_seconds` got worse by about `10.633s`

That regression is substantially larger than the measured baseline noise band, so this was a real slowdown rather than random variance.

## Profiled After-Run Evidence

One profiled after run produced:

- `duration_seconds=183.036`
- `battle_loop_seconds=177.151`

Damage-cache stats from the instrumented profile:

- `hits=1,188`
- `misses=83,758`
- `evictions=67,374`
- `size=16,384`

Interpretation:

- hit rate was only about `1.4%`
- the conservative key-building overhead and cache churn dominated any savings from those hits
- this confirms that naive request-local exact damage memoization is **not** the right next optimization in this codepath

Updated priority adjustment:

1. do **not** pursue broad exact-damage memoization with heavy battle-state keys
2. instead focus on reducing the number of exact damage calls entirely
3. the next strongest direction is active-board-only exactness plus cheaper bench summaries
4. lower-risk stat-bound caching remains worthwhile because it is cheap to implement and has much less overhead risk

# Follow-Up Experiment: Active-Board-Only Exactness With Cheap Bench Estimates

Tested the next proposed direction by keeping exact damage only when both mons in the engineered feature pair were currently active, while replacing other damage estimates with a cheap approximation based on:

- move base power
- move category
- rough offensive and defensive stats
- STAB
- type effectiveness
- spread-target penalty

This preserved feature shape but intentionally reduced exact bench damage work.

## After Implementing Active-Board-Only Exactness (`3` runs)

Observed after-change variance:

- `duration_seconds`: `195.249`, `163.758`, `181.441` (mean `180.149`)
- `battle_loop_seconds`: `189.382`, `157.878`, `175.578` (mean `174.279`)

Relative to the same baseline mean:

- `duration_seconds` got worse by about `82.675s`
- `battle_loop_seconds` got worse by about `82.639s`

This is far beyond the measured baseline noise band, so the experiment was a severe regression.

## Profiled After-Run Evidence

One profiled after run produced:

- `benchmark_seconds=390.317`
- `Embedder.embed_to_array = 42.731s` over `5,829` calls (`7.33ms` average)
- `Embedder.generate_feature_engineered_features = 14.249s` over `5,838` calls (`2.44ms` average)
- `embedder.calculate_damage = 5.195s` over `24,104` calls
- `embedder.compute_stats = 1.705s` over `94,052` calls

Critical interpretation:

- the engineered feature block really did get cheaper per call
- exact damage calls dropped sharply (`79,610` down to `24,104` in the profiled run)
- but the policy quality degraded enough that battles became dramatically longer
- the embedder saved work locally, but the cheaper approximation changed action quality enough to destroy end-to-end throughput

Updated priority adjustment after this second test:

1. broad exact-damage memoization is not viable in this path as tested
2. aggressive active-board-only exactness with cheap bench estimates is also not viable as tested
3. future damage-work reductions need to preserve much more of the strategic bench signal than this approximation did
4. the next safer optimization candidate is low-overhead stat-bound caching or narrower exact-damage skipping for clearly non-damaging moves