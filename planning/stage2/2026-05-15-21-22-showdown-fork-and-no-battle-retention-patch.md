# Showdown fork + `--no-battle-retention` patch

**Date**: 2026-05-15 21:22
**Trigger**: `radiant-violet-57` and follow-up may15.yaml runs tripped
the memory watchdog because the Showdown server's resident-set kept
climbing (~1.3 GB/hr) over multi-hour training. Option A (sending
`/leavebattle` on poke-env's `_recover_room_lost_battle` popup path)
did not reduce growth. The user chose Option B: patch Pokémon Showdown
directly so finished battles drop immediately.

## Context

Default Showdown room retention is set in `server/rooms.ts`:

```ts
const TIMEOUT_EMPTY_DEALLOCATE = 10 * 60 * 1000;     // 10 min
const TIMEOUT_INACTIVE_DEALLOCATE = 40 * 60 * 1000;  // 40 min
```

`Room.pokeExpireTimer()` is called from `Room.update()` whenever the
broadcast buffer flushes. After `|win|`/`|tie|`, the GameRoom sits
around 10–40 min holding battle log / inputLog / replay-data state.

In RL training we finish hundreds of battles per minute and the AI
clients don't always promptly leave the rooms (the `_battle_locks`
issue patched 2026-05-11 reduced leaks but didn't eliminate the
retention window). That accumulation is the dominant Showdown memory
growth source.

## Before State

### `../pokemon-showdown` local clone (pre-session)

- `origin` → `smogon/pokemon-showdown` (no fork on GitHub)
- Local `master` at `9fae1b8a0` — 421 commits behind actual upstream;
  `git status` reported "up to date" because the local cache was stale.
- Working tree: a transient `package-lock.json` bump from a prior
  `npm install` (unrelated to RL work).
- No custom patches in the working tree. `--no-security` is already
  built into upstream (`server/config-loader.ts:35`) via `FLAG_PRESETS`
  — it maps to `nothrottle,noguestsecurity,noipchecks`.

### `RoomBattle.end()` in `server/room-battle.ts`

After the simulator emits `'end'`:

1. `this.room.update()` flushes the `|win|`/`|tie|` broadcast and (via
   `pokeExpireTimer`) schedules destroy in 10–40 min.
2. Optional `this.room.uploadReplay(...)` if `replaySaved` or
   `Config.autosavereplays` is set.
3. Method returns — room then sits idle until the timer fires.

### `showdown_server_manager.py` launch command

```python
["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)]
```

## Problem

Showdown RSS climbs ~1.3 GB/hr in RL training because finished
GameRooms accrue for 10–40 min each. With Plan C's higher throughput,
this approaches the WSL2 23 GB physical RAM ceiling within ~6 hours and
trips the memory watchdog. Option A (force-`/leavebattle` from
poke-env) did not reduce growth in v5: server rejects `/leavebattle` to
rooms the user has already been kicked from, and even prompt
disconnects only flip empty→inactive timer states without dropping the
room.

## Solution

### Showdown patch (on fork branch `elitefurretai-patches`)

Two-file, 20-line patch following the `--no-security` precedent:

**`server/config-loader.ts`** — new CLI alias:

```ts
['--no-battle-retention', ['nobattleretention']],
```

**`server/room-battle.ts`** — in `RoomBattle.end()`, after broadcast
and replay handling:

```ts
if (this.replaySaved || Config.autosavereplays) {
    const uploadPromise = this.room.uploadReplay(...);
    if (Config.nobattleretention) {
        void uploadPromise.then(
            () => this.room?.destroy(),
            () => this.room?.destroy(),
        );
    }
    return uploadPromise;
}
if (Config.nobattleretention) {
    setImmediate(() => this.room?.destroy());
}
```

Race-safety details:

- `setImmediate` ensures the broadcast queued by `this.room.update()`
  reaches sockets before deallocation.
- The replay branch waits on `uploadPromise` settling (resolve *or*
  reject) so the async `getLog()` call inside `uploadReplay` doesn't
  race with `this.log.destroy()`.
- `this.room?.destroy()` is optionally-chained because
  `RoomBattle.destroy()` nulls `this.room` (`room-battle.ts:1272`), so
  if any other path destroyed the room first the deferred callback is
  a no-op.

### Launcher wire-up

`src/elitefurretai/engine/showdown_server_manager.py:55` — add
`"--no-battle-retention"` to the `node pokemon-showdown start ...`
argv. No other call-sites need changes; `--no-security` and
`--no-battle-retention` are independent.

### Fork topology

- `caymansimpson/pokemon-showdown` (GitHub) — created via `gh repo fork`
- Local remotes: `origin` → fork, `upstream` → smogon
- Local `master` fast-forwarded 421 commits to `10f9a5c0d`
- Patch branch `elitefurretai-patches` from new master tip;
  commit `b6d57beb2` pushed to `origin/elitefurretai-patches`

## Reasoning

**Why a CLI flag and not a code path edit?**
Mirrors `--no-security`: explicit opt-in, default-off, no risk to any
non-AI Showdown deployment. Production servers retain existing
behavior. Lets us keep the fork's diff minimal so rebasing on future
upstream changes stays cheap.

**Why immediate destroy vs shortening the timers to e.g. 30s?**
Lower-blast-radius patch: a single `setImmediate(() => destroy())` at
exactly one well-defined event (`RoomBattle.end()`) is easier to reason
about than dialing `TIMEOUT_EMPTY_DEALLOCATE`/`TIMEOUT_INACTIVE_DEALLOCATE`
globally, which would also affect chat rooms and other room types.

**Why not destroy synchronously inside `end()`?**
Two reasons:
1. `this.room.update()` (called earlier in `end()`) has just queued
   bytes for socket write. Async I/O happens on the next tick.
2. `RoomBattle.destroy()` and `Room.destroy()` mutate the room's own
   state (null out `this.room`, deregister, destroy log). Doing this
   inside the same `end()` invocation that called
   `this.room.parent?.game?.onBattleWin?.(this.room, winnerid)` etc.
   could create dangling references. Deferring to next tick keeps the
   teardown atomic.

**Why update both showdown and launcher in lockstep?**
The flag is no-op without the launcher passing it; the launcher arg is
no-op without showdown understanding it. They're coupled. Bundling
the commits in this repo with the planning doc keeps the change story
together.

## Planned Next Steps

1. **Relaunch may15.yaml** with the patched showdown and updated
   launcher. Validate that showdown RSS stays roughly flat (or grows
   ≪ 1.3 GB/hr) instead of climbing. Watch ~1–2 hours for confidence.
2. **Commit launcher change + this planning doc** together on EliteFurretAI
   `main` once validation confirms behavior.
3. **If validation fails**: fall back to investigating whether log
   files / replay state stay anchored elsewhere (e.g. `Rooms.rooms`
   registry, replay queue, parent tour state). Possible follow-ups:
   - Force-shrink `TIMEOUT_EMPTY_DEALLOCATE` to 30s as a belt-and-braces.
   - Add diagnostic logging of `Rooms.rooms.size` in the simulator
     subprocess.
4. **Memory watchdog**: with growth controlled, consider dropping
   `memory_watchdog_threshold_gb` back from 24 → 22 in may15.yaml so it
   resumes being a real safety net rather than just a ceiling.

## Updates

*(none yet — to be filled in after validation run)*
