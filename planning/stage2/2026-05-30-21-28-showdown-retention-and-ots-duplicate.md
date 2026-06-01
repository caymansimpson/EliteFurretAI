# Showdown no-battle-retention desync fix + OTS forme-duplicate findings

**Date**: 2026-05-30
**Status**: Showdown patch shipped + committed; vgc-bench OTS crash patched (embed-level);
poke-env root-cause fix deferred.

Two related investigations that came out of adding the Open Team Sheets (OTS) config
flag and running vgc_bench evals.

## 1. Showdown `--no-battle-retention` → "room does not exist" desync

### Context / Before
EFA's Showdown servers (launched by `engine/showdown_server_manager.py:launch_showdown_servers`)
always pass `--no-battle-retention`, a local patch in our pokemon-showdown fork
(`caymansimpson/pokemon-showdown`, commit `b6d57beb2`) that destroys finished battle
rooms instead of retaining them 10–40 min (saves ~1.3 GB/hr RSS in high-throughput RL).

### Problem
The patch destroyed the room via `setImmediate(() => room.destroy())` (or immediately
after the replay upload settled) right after the `|win|/|tie|` broadcast. poke-env reads
the battle over the websocket from a **separate process**; this races it. When `destroy()`
won the race, the room vanished before poke-env finished processing the battle-end
messages, producing:
- a flood of `/error The room 'battle-…' does not exist` PMs, and
- intermittent **hangs**: the battle never registered as finished, so the in-process
  side's `send_challenges(n_challenges=…)` awaited forever.

Symptoms matched: intermittent (a race), branch-independent (server-side, hit under both
poke-env fork branches), and worst for `simple_heuristic` (longer games shift timing so
destroy more often wins). Confirmed root cause by reading the patch; the flood + hang
reproduced on `simple_heuristic` constrained-team evals.

### Fix (shipped)
`server/room-battle.ts`: replaced `setImmediate`/immediate-on-upload teardown with a 2 s
timer, `const NO_RETENTION_DESTROY_DELAY_MS = 2 * SECONDS`. Long enough for out-of-process
clients to finish the `|win|/|tie|` handling and leave; still frees rooms far sooner than
the 10–40 min default. Built (`node build`, picked up automatically by the `pokemon-showdown`
launcher) and committed: `caymansimpson/pokemon-showdown` branch `elitefurretai-patches`,
commit `3f722a7f0` (on top of `b6d57beb2`). Not yet validated live (deferred).

## 2. OTS forme-duplicate crash in vgc_bench

### Problem (confirmed empirically)
With Open Team Sheets accepted, the vgc_bench `PolicyPlayer` crashes on its first forward
pass: `RuntimeError: shape '[1, 12, -1]' is invalid for input of size 8670`. `embed_battle`
builds one chunk per `battle.team` / `battle.opponent_team` entry (pad-up-to-6, **no cap**),
and the policy reshapes assuming exactly 12 (`x.view(batch, 12, -1)`).

Logged `battle.team` during a live OTS battle: the bot's own team had **9** entries — the
3 forme Pokémon each appear twice, under a base-name key and a forme-name key:

```
('p2: Calyrex', 'calyrexice')          ('p2: Calyrex-Ice', 'calyrexice')
('p2: Urshifu', 'urshifurapidstrike')  ('p2: Urshifu-Rapid-Strike', 'urshifurapidstrike')
('p2: Ogerpon', 'ogerponhearthflame')  ('p2: Ogerpon-Hearthflame', 'ogerponhearthflame')
```

9 own + 6 opp = 15 chunks = 8670.

### Root cause (poke-env fork)
The OTS `showteam` handler (`player/player.py`) re-registers each revealed mon via
`get_pokemon(f"{role}: {teambuilder_nickname}", details=…)` — keyed by the **forme name** —
while team-preview/`|request|` registered it under the **base** species name. At `showteam`
time the existing entry's species is still the base (`calyrex`), so `get_pokemon`'s de-dupe
(`identifies_as`) misses and inserts a second entry. Three formes → three duplicates.

### Does EFA's own poke-env fork (`caymansimpson/poke-env`) have it?
The `showteam` handler is **byte-identical** across both forks. EFA's fork has a *hardened*
`get_pokemon` de-dupe (3 match conditions vs cameronangliss's 1), but on analysis those
conditions still don't cover this case (base-key entry is base-species at `showteam` time;
forme-name `identifies_as` and the stored `_name` both miss). So EFA's fork **very likely
populates the same duplicate** under OTS. It would **not** crash like vgc_bench, though —
EFA agents use EFA's embedder, not vgc-bench's hard-coded `view(batch, 12, -1)`. Only
relevant if EFA enables `open_team_sheets` for its own agents; an empirical OTS run of an
EFA agent would confirm whether EFA's embedder mis-embeds the inflated team. **Unconfirmed
empirically** (no validation run done).

### Fix applied (vgc-bench embed-level, certain)
Patched `vgc-bench-src-bcsp/vgc_bench/src/policy_player.py` `embed_battle` to de-dupe each
side by species (Species Clause ⇒ one real mon per species), keeping the first
(request-registered, battle-active) entry, then cap to 6. This guarantees 12 chunks and
keeps active-slot detection correct (the kept entry is the active object). Local patch on
the editable `vgc-bench-src-bcsp` clone. **Not validated live** (deferred per request).

Chose the embed-level fix over editing the poke-env fork because the correct poke-env
internals fix (make `showteam` update the existing team-preview object instead of
re-`get_pokemon`-ing a forme-name identifier) needs live iteration to get right, whereas
the embed de-dupe+cap is robustly correct without a run. The poke-env root-cause fix
(and upstreaming both to the author) remains a follow-up.

## 3. OTS config params surfaced in may26.yaml
Added `open_team_sheets: false` at top level (training) and under `eval:` in
`configs/may26.yaml`, with comments — both are the defaults, surfaced for discoverability,
and noting vgc_bench can't run with OTS on.

## Planned next steps
- Validate the Showdown 2 s-timer fix (re-run the `simple_heuristic` matchup; expect the
  "room does not exist" flood + hangs to drop).
- Optionally implement + live-test the poke-env-fork root fix and upstream both the
  showteam-dedup and the vgc-bench embed cap to the author.
- Decide whether to merge the `feat/open-team-sheets-config` branch.

## Updates
_(none yet)_
