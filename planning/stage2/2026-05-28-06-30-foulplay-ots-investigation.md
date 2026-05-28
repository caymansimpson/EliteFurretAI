# FoulPlay smoke test + OTS investigation

## Context

Attempted to run a foul-play smoke as a sanity check for the RL eval
method. Goal: a single battle of `max_damage` vs `foul_play` on
`gen9vgc2024regg` to confirm the orchestration in
`src/elitefurretai/rl/analyze/evaluate.py` completes end-to-end.

This session pulled on the OTS thread because the original `foul_play`
subprocess assumed `|showteam|` lines at teampreview, which Showdown
only emits under Open Team Sheets. The investigation surfaced three
distinct issues, only one of which is fully solved.

## Before State

Session started with `M src/elitefurretai/agents/_foulplay_subprocess.py`
and `M src/elitefurretai/rl/analyze/evaluate.py` already in the working
tree. The pre-session subprocess version had a monkey-patch of
`fp.run_battle.start_standard_battle` that tolerated a missing
`|showteam|` line ("graceful fallback" branch). Pre-session
`evaluate.py` had `_opponent_requires_ots` always returning False with a
TODO noting that OTS plumbing was kept "for future opponents."

Eval plumbing itself (server boot, foul-play subprocess launch,
`launch_external_player`, win-rate accounting) was already validated by
prior work (see `2026-05-25-23-37-foulplay-eval-scope-confirmed.md`).

## Problem

Three layered failures stack up as a smoke proceeds:

1. **`SmogonSets._get_pokemon_information` crashes.** Once a battle
   reaches teampreview, foul-play reads its cached Smogon stats JSON at
   `foul-play-doubles/data/smogon_stats_cache/gen9vgc2024regg-0.json`
   and crashes on
   `round(counter_information["p"], 2)` in
   `foul-play-doubles/data/pkmn_sets.py:238`. The cache stores
   `"Checks and Counters"` values as arrays `[n, p, d]` (the standard
   Smogon-chaos format), not dicts.

2. **Missing `|showteam|` empties `opponent.reserve` → empty MCTS
   policy.** Foul-play's `start_standard_battle` populates
   `battle.opponent` from a `|showteam|<opp_name>|<packed_string>`
   line via `from_packed_string`. Showdown only emits `|showteam|`
   when the format has the `Open Team Sheets` rule active **and** both
   players send `/acceptopenteamsheets`. The pre-session graceful
   fallback skipped the parse but left `opponent.reserve = []`, so
   `get_battles_for_team_preview` had nothing to expand and MCTS
   returned an empty policy. Crash at
   `fp/search/main.py:129 final_policy[0][1]` IndexError.

3. **Mystery regression: any subprocess monkey-patch prevents
   `/challenge` delivery.** This is the unsolved one. Every attempted
   change to `_foulplay_subprocess.py` that defines and installs a
   replacement for `fp.run_battle.start_standard_battle` (even a
   syntactically minimal one) caused `max_damage`'s `/challenge` PM to
   never reach foul-play. Foul-play sits in `accept_challenge`'s
   "Waiting for a gen9vgc2024regg challenge" loop indefinitely. With
   the same `_foulplay_subprocess.py` reverted to HEAD (no
   monkey-patch), challenges reach foul-play immediately. Directly
   editing `foul-play-doubles/fp/run_battle.py` (instead of
   monkey-patching) reproduces the same hang.

## Solution

Issue 1 has a one-line fix:
`counter_information["p"]` → `counter_information[1]` at
`foul-play-doubles/data/pkmn_sets.py:238`. The `[1]` index gives the
percentage (`p`) in the standard Smogon-chaos format. Applied during
the session and verified by re-running the smoke (foul-play got past
SmogonSets initialization).

Issue 2 has a known protocol path:

- Showdown's `[Gen 9] VGC 2024 Reg G` format (and Reg I, Reg F) lists
  `Open Team Sheets` in its ruleset (`pokemon-showdown/config/formats.ts:233`).
- The `openteamsheets` rule emits a `|uhtml|otsrequest|...` button at
  teampreview; the `forceopenteamsheets` rule emits `|showteam|`
  unconditionally (`pokemon-showdown/data/rulesets.ts:1979` and 2001).
- When `openteamsheets` is in effect, both players must send
  `/acceptopenteamsheets` (poke-env does this automatically when its
  `accept_open_team_sheet=True` flag is set). Once both `wantsOpenTeamSheets`
  flags are true, Showdown writes `>show-openteamsheets` to the battle
  stream, which calls `battle.showOpenTeamSheets()` and broadcasts
  `|showteam|p1|...` and `|showteam|p2|...`
  (`pokemon-showdown/sim/battle.ts:3205`).
- `|teampreview|` and `|showteam|` arrive in *separate* WS frames,
  so foul-play needs to keep reading past the first teampreview frame.

The right shape for the foul-play side is therefore:

1. Read until `|teampreview|` lands.
2. If `|uhtml|otsrequest` is present, send `/acceptopenteamsheets <room>`.
3. Keep buffering frames until `|showteam|` arrives.
4. Parse `|showteam|p<N>|<packed>` and feed `from_packed_string`.

Issue 3 is unsolved. All four variants tried (eager `/accept` after
`start_battle_common`, reactive `/accept` on otsrequest, in-subprocess
monkey-patch, direct edit of `fp/run_battle.py`) reproduce the same
hang where max_damage's challenge PM never reaches foul-play's
`accept_challenge` loop. The monkey-patch only modifies code that
runs *after* a battle starts, so the mechanism is not obvious. Bisect
narrows it down: reverting only the subprocess monkey-patch (keeping
the eval-side OTS wiring) restores challenge delivery.

## Reasoning

**Why patch foul-play-doubles directly vs monkey-patch from EFA.** Per
prior session preference ("Patch foul-play-doubles directly" was the
chosen approach for issue 1), a direct edit is the cleanest path: no
runtime monkey-patch, no Python import-order subtleties, and the diff
shows up cleanly in `foul-play-doubles`'s git status. The mystery
regression in issue 3 makes this preference even stronger, since both
the monkey-patch and the direct edit reproduce the hang — but at least
with a direct edit the diff is visible to whoever's debugging.

**Why OTS is the right path for VGC eval.** Foul-play's MCTS needs the
opponent's full team (moves, ability, item, tera-type, EVs) to expand
nodes at teampreview. The `|poke|` lines Showdown emits at teampreview
carry species + a has-item flag only — not enough to build a
`PokemonSpread`. Sampling unknown moves from `SmogonSets` is possible
in principle but no code path in foul-play-doubles wires that up at
teampreview (only `sample_unrevealed_pkmn` does it mid-battle, and only
for species already present in `opponent.reserve`). OTS is the cheapest
way to get a populated `opponent.reserve` at teampreview. Real
tournament play (Worlds, regionals) uses OTS for VGC reg G/H, so this
matches realistic eval conditions.

**Why the SmogonSets `[1]` fix is real.** The Smogon-chaos JSON format
specifies `Checks and Counters` values as `[n, p, d]` arrays
(`n` = sample, `p` = KO+switch percentage, `d` = std dev). The cached
file at `foul-play-doubles/data/smogon_stats_cache/gen9vgc2024regg-0.json`
follows this. `_get_pokemon_information` accesses
`counter_information["p"]` which would only work against a transformed
shape that foul-play-doubles never produces. The crash only manifests
when a teampreview-mon name overlaps with another mon's
"Checks and Counters" list, which is rare in lower-tier formats but
constant in VGC because the species pool is small.

## Planned Next Steps

Pick up in a fresh session with no monkey-patch artifacts in scope.
Suggested order:

1. **Re-apply the SmogonSets `[1]` fix.** Direct edit to
   `foul-play-doubles/data/pkmn_sets.py:238`:
   `counter_information["p"]` → `counter_information[1]`. This is
   independent of OTS and unblocks the next layer.

2. **Diagnose the monkey-patch hang first, before any further OTS
   work.** Suggested experiment: take the current bare
   `_foulplay_subprocess.py` (no patch) and add the *minimum possible*
   change — e.g. a top-level `import fp.run_battle` and a single
   `fp.run_battle.start_standard_battle = fp.run_battle.start_standard_battle`
   (a no-op self-assignment) — and verify challenge delivery still
   works. Then add steps incrementally:
   - Define `_patched = fp.run_battle.start_standard_battle` (alias)
   - Assign `fp.run_battle.start_standard_battle = _patched` (round-trip)
   - Define a wrapper that just `await`s the original
   - Add the actual `|showteam|` buffer logic
   Each step should reproduce the bug at exactly one boundary. The
   bisect should narrow whether it's the import-order shift, the
   attribute reassignment, the wrapper function's coroutine identity,
   or something else.

3. **Once challenge delivery is robust, apply the OTS handshake.**
   Direct edit `foul-play-doubles/fp/run_battle.py:223` to:
   - Wait for `|teampreview|`.
   - If `|uhtml|otsrequest` is in the buffer, send
     `/acceptopenteamsheets <room>`, then buffer further frames until
     `|showteam|` arrives.
   - Parse `|showteam|p<N>|<packed>` into `opponent.from_packed_string`.
   - Add a fallback (timeout or explicit reject signal) so the
     subprocess doesn't hang forever if Showdown never broadcasts
     `|showteam|`.

4. **Flip `_opponent_requires_ots` in evaluate.py to True for
   `foul_play`.** This propagates `accept_open_team_sheet=True` into
   the in-process poke-env Player so it sends `/acceptopenteamsheets`
   on battle init.

5. **Run the 100-battle smoke** against `max_damage` and (separately)
   against `vgc_bench` in `gen9vgc2024regg` once the above completes
   a single battle cleanly.

## Updates

(none)
