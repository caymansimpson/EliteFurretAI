# RL Training Debug Learnings

This document captures the bugs discovered and fixed while debugging `rl/train.py` to work with `easy_test.yaml`.

## Summary of Issues Fixed

| Issue | File | Root Cause | Fix |
|-------|------|------------|-----|
| Switch encoding mismatch | `etl/encoder.py` | `battle.team` dict order ≠ Showdown request order | Use `battle.last_request["side"]["pokemon"]` for switch targets |
| Struggle/recharge moves | `etl/encoder.py` | These moves aren't in Pokemon's learned moveset | Check `available_moves[i]` for struggle/recharge and use directly |
| Double force_switch with 1 mon | `etl/battle_order_validator.py` | Pass was invalid even when no switches available | Allow pass when `force_switch=[True,True]` and only 1 mon left |
| Trajectory length overflow | `rl/train.py` | Battles can have 39+ steps with all force_switches | Truncate trajectories to `max_seq_len=17` |

---

## Issue 1: Switch Encoding Mismatch

### Symptom
```
|error|[Invalid choice] Can't switch: The Pokémon "Growlithe" can only switch in for a Pokémon that has already been sent out.
```

### Root Cause
When encoding `switch N` (where N is 1-4), MDBO was using `list(battle.team.values())[N-1]` to get the Pokemon. However:

1. `battle.team` is a Python dict populated during **teampreview** with all 6 Pokemon
2. After teampreview, Showdown **reorders** the request's `side.pokemon` list so positions 1-4 are the Pokemon actually brought to battle
3. Python dicts maintain insertion order, so `battle.team` still has the original teampreview order
4. This caused `switch 1` to sometimes select a Pokemon that wasn't even brought to battle

### Fix (encoder.py)
```python
# Before: Used battle.team dict order (WRONG)
team_list = list(battle.team.values())
mon = team_list[switch_idx]

# After: Use request's side.pokemon order (CORRECT)
if battle.last_request and "side" in battle.last_request:
    request_pokemon = battle.last_request["side"]["pokemon"]
    if switch_idx < len(request_pokemon):
        ident = request_pokemon[switch_idx]["ident"]
        if ident in battle.team:
            mon = battle.team[ident]
```

### Why This Was Subtle
- Only manifests when teampreview selection differs from default order
- Self-play with random teams triggers this frequently
- Supervised learning from logs doesn't have this issue (logs already reflect final ordering)

---

## Issue 2: Struggle/Recharge Moves

### Symptom
```
WARNING: No valid actions found for battle...
Available moves: [[struggle], [protect, ...]]
```

### Root Cause
When a Pokemon runs out of PP on all moves, Showdown allows only `struggle`. However:

1. `struggle` is **not** in the Pokemon's learned `moves` dict
2. MDBO's `move N` encoding looked up `moving_mon.moves[N-1]`
3. This returned some other move that wasn't actually available
4. Validation failed because the looked-up move wasn't in `available_moves`

Same issue applies to `recharge` (used after Hyper Beam, etc.)

### Fix (encoder.py)
```python
# Check if only struggle/recharge is available
if (len(battle.available_moves[i]) == 1
    and battle.available_moves[i][0].id in ["struggle", "recharge"]):
    # Use the move from available_moves directly
    move = battle.available_moves[i][0]
else:
    # Normal case: look up from Pokemon's moveset
    move_idx = int(order[5]) - 1
    move_key = list(moving_mon.moves.keys())[move_idx]
    move = moving_mon.moves[move_key]
```

### Reference
poke-env's `DoublesEnv` has the same special case handling in its action-to-order conversion.

---

## Issue 3: Double Force_Switch with Limited Pokemon

### Symptom
```
WARNING: No valid actions found...
Force switch: [True, True], Active: [None, None]
Available switches: [[dragonite], [dragonite]]
```

### Root Cause
When both active Pokemon faint but only 1 replacement is available:
1. `force_switch=[True, True]` - both slots need to switch
2. But there's only 1 Pokemon left
3. One slot switches, the other **must pass**
4. The validator was rejecting pass for any slot where `force_switch[i]=True`

### Fix (battle_order_validator.py)
```python
if order is None or isinstance(order, PassBattleOrder):
    if battle.force_switch[i]:
        # Exception: both slots need to switch but not enough mons
        if battle.force_switch[0] and battle.force_switch[1]:
            all_available = set()
            for slot_switches in battle.available_switches:
                for mon in slot_switches:
                    all_available.add(mon.species)
            # If only 1 available switch, one slot MUST pass
            if len(all_available) <= 1:
                pass  # Allow pass in this case
            else:
                return False
        else:
            return False
```

---

## Issue 4: Trajectory Length Overflow

### Symptom
```
ValueError: Sequence length 39 exceeds max_seq_len 17
```

### Root Cause
The supervised model was trained with `max_seq_len=17` (positional embedding size), assuming VGC battles have at most 17 decision points:
- 1 teampreview
- ~16 turns maximum

However, with force_switch counting as separate decisions:
- Each turn can have 1-3 decisions (normal turn + 0-2 force switches)
- A long battle can easily have 39+ decision points

### Fix (train.py)
```python
def collate_trajectories(trajectories, device, max_seq_len=17):
    # Truncate long trajectories to max_seq_len
    truncated_trajectories = []
    for traj in trajectories:
        if len(traj) > max_seq_len:
            # Keep last max_seq_len steps (later decisions matter more)
            truncated_trajectories.append(traj[-max_seq_len:])
        else:
            truncated_trajectories.append(traj)
```

### Future Improvement
For production, consider:
1. Re-training model with larger `max_seq_len` (e.g., 50)
2. Using relative positional encoding (no max length limit)
3. Rolling window approach for very long battles

---

## Key Technical Insights

### battle.team vs battle.last_request
- `battle.team`: Dict of Pokemon objects, keyed by identifier (e.g., "p1: Iron Hands")
  - Populated during teampreview
  - Order reflects teampreview order, NOT current battle order
  - Contains all 6 Pokemon

- `battle.last_request["side"]["pokemon"]`: List from Showdown's request
  - Order reflects current battle state
  - After teampreview, positions 1-4 are the brought Pokemon
  - **Use this for switch index mapping!**

### MDBO Action Space
- 45 single actions: 40 moves (4 moves × 5 targets × 2 tera states) + 4 switches + 1 pass
- 2,025 combined actions: 45 × 45 for both active Pokemon
- Switch indices 40-43 map to `switch 1-4` in Showdown protocol
- `switch N` refers to position N in request's `side.pokemon` list, NOT `battle.team` order

### Force Switch States
| force_switch | Available Switches | Valid Actions |
|--------------|-------------------|---------------|
| `[False, True]` | 2+ for slot 1 | `pass, switch N` |
| `[True, False]` | 2+ for slot 0 | `switch N, pass` |
| `[True, True]` | 2+ total | `switch N, switch M` (N≠M) |
| `[True, True]` | 1 only | `switch N, pass` OR `pass, switch N` |

### Struggle/Recharge
These are "forced" moves that appear when normal moves are unavailable:
- `struggle`: All moves have 0 PP
- `recharge`: After using Hyper Beam, Giga Impact, etc.

Both appear in `battle.available_moves[i]` but NOT in `pokemon.moves` dict.

---

## Debug Tips

### Useful Debug Output Pattern
```python
if any(battle.force_switch):
    valid_actions = np.where(mask > 0)[0][:10]
    active_status = []
    for i, mon in enumerate(battle.active_pokemon):
        if mon is None:
            active_status.append(f"slot{i}=None")
        else:
            active_status.append(f"slot{i}={mon.species},fainted={mon.fainted}")
    print(f"DEBUG FORCE_SWITCH: force_switch={battle.force_switch}, "
          f"action={action_idx}, order={order}, "
          f"valid_actions={valid_actions}, active_status={active_status}")
```

### Checking Request vs Team Order
```python
# Print request pokemon order
if battle.last_request and "side" in battle.last_request:
    request_order = [p["ident"] for p in battle.last_request["side"]["pokemon"]]
    print(f"Request order: {request_order}")

# Print team dict order
team_order = list(battle.team.keys())
print(f"Team dict order: {team_order}")
```

### Validating Action Masks
When debugging mask issues:
```python
# Check which actions pass validation
for action_idx in range(2025):
    mdbo = MDBO.from_int(action_idx, type=action_type)
    try:
        order = mdbo.to_double_battle_order(battle)
        valid = is_valid_order(order, battle)
        if valid:
            print(f"Action {action_idx}: {mdbo.message} -> VALID")
    except Exception as e:
        print(f"Action {action_idx}: {mdbo.message} -> ERROR: {e}")
```

---

## Files Modified

1. **`src/elitefurretai/etl/encoder.py`**
   - Fixed switch target lookup to use request ordering
   - Added struggle/recharge special case handling

2. **`src/elitefurretai/etl/battle_order_validator.py`**
   - Fixed pass validation for double force_switch with limited Pokemon

3. **`src/elitefurretai/rl/train.py`**
   - Added trajectory truncation in `collate_trajectories()`

4. **`src/elitefurretai/rl/worker.py`**
   - Removed DEBUG_FORCE_SWITCH logging (after fix verified)
