# Mega-Format ETL→RL Support (Plans 2–4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ETL→RL pipeline represent mega-evolution end-to-end — embedder features (including the prospective mega form), entity vocab, mega-aware heuristic opponents, and a regMA-capable curriculum — so one agent can train and be evaluated across tera and mega formats.

**Architecture:** Three sequenced parts. **Part A (Plan 2, embedder)** is the gate: it syncs poke-env data, adds Mega Stones to the item vocab, adds a prospective mega-form feature block (mirroring `TERA_TYPE` but fuller and multi-hot), adds mega scalar flags, then re-processes tera tensors and re-finetunes BC with a vocab-aware warm-start. **Part B (Plan 3, opponents)** makes the heuristic baselines play mega; it is independent and only needs the already-shipped action layer. **Part C (Plan 4)** adds regMA to the curriculum and validates the engine end-to-end.

**Tech Stack:** Python, poke-env (local fork at `/home/cayman/Repositories/poke-env`), PyTorch, numpy, pytest, Zstandard-compressed tensors.

**Reference design:** `planning/stage2/2026-05-27-17-13-mega-format-etl-rl-support-design.md`. Action layer (Plan 1) already shipped on `main`.

**Environment:** Always `source ../venv/bin/activate` from repo root before python/pytest. Quality gates after non-trivial edits: `ruff check src unit_tests`, `ruff format src unit_tests --check`, `pyright <changed files>`, `pytest unit_tests -q`.

**Task types:** Tasks marked **[TDD]** follow the failing-test-first cycle. Tasks marked **[OPS]** are operational (git merge, data processing, training, live-server validation) that cannot be unit-TDD'd; they give exact commands and explicit acceptance criteria instead.

---

## Grounding facts (verified 2026-05-27)

- gen9 `GenData.pokedex` already contains all 101 mega formes. Each carries `requiredItem` (the Mega Stone, e.g. `"Venusaurite"`, `"Charizardite X"`), `baseStats`, `types` (1–2), `abilities` (`{"0": ...}`), `baseSpecies`, and `forme` (`"Mega"`/`"Mega-X"`/`"Mega-Y"`). Rayquaza-Mega uses `requiredMove` not a stone (excluded; banned in reg formats).
- `build_species_to_id/ability/move(gen)` ([embedder.py:1774-1807](../../src/elitefurretai/etl/embedder.py)) enumerate the full gen9 dex/movedex → species/abilities/moves auto-covered if poke-env data is current.
- `TRACKED_ITEMS` ([embedder.py:1668](../../src/elitefurretai/etl/embedder.py)) is curated and has no Mega Stones; `ITEM_TO_ID` is built from it ([embedder.py:1771](../../src/elitefurretai/etl/embedder.py)).
- Per-mon features are built in `generate_pokemon_features` (own, [embedder.py:899](../../src/elitefurretai/etl/embedder.py)) and `generate_opponent_pokemon_features` ([embedder.py:1052](../../src/elitefurretai/etl/embedder.py)); battle features in `generate_battle_features` ([embedder.py:1195](../../src/elitefurretai/etl/embedder.py)). The `TYPE:` block is multi-hot ([line 968](../../src/elitefurretai/etl/embedder.py)); `TERA_TYPE` is single-hot.
- Stats utility: `compute_stats(mon, "max"|"min")` ([embedder.py:1535](../../src/elitefurretai/etl/embedder.py)) wraps `compute_raw_stats`.
- `embedding_size = len(self.embed(dummy_battle))` ([embedder.py:118](../../src/elitefurretai/etl/embedder.py)); model input dim = `embedder.embedding_size` ([model_archs.py:1011](../../src/elitefurretai/supervised/model_archs.py)). Checkpoint load supports `strict=False` ([learners.py:882](../../src/elitefurretai/rl/learners.py)).
- poke-env fork: `upstream` = `hsahovic/poke-env`; data refreshes arrive as upstream "JSON data update from smogon" commits. Fork carries EFAI custom commits (ObservedPokemon, observations, DoubleBattle team-order sync) and has uncommitted `player.py`/`ps_client.py` edits.
- `MEGA_TYPE` must be **multi-hot** (mega keeps/changes dual typing), not single-hot.

## File Structure

- Modify: `src/elitefurretai/etl/embedder.py` — `build_mega_stone_to_species` helper + `MEGA_STONE_TO_SPECIES`/item-vocab union (Task A1); `_prospective_mega_entry` + prospective block in both per-mon methods (Task A2); mega scalar flags in per-mon + battle features (Task A3).
- Modify: `src/elitefurretai/etl/battle_iterator.py` — track the `|-mega|` event to expose a per-mon mega-evolved signal (Task A3, pending verification).
- Modify: `src/elitefurretai/agents/max_damage_player.py`, `src/elitefurretai/rl/opponents.py` (or the simple-heuristic wrapper) — mega-aware enumeration (Task B1).
- Modify: `src/elitefurretai/rl/config.py` + the active curriculum yaml — regMA sampling + tera:mega ratio knob (Task C1).
- Modify: `src/elitefurretai/rl/learners.py` (or supervised load util) — vocab-aware warm-start transfer (Task A5).
- Tests: `unit_tests/etl/test_embedder.py`, `unit_tests/etl/test_embedder_improvements.py`, `unit_tests/rl/test_opponent_pool.py`, plus new test functions.
- Poke-env fork repo: data sync only (Task A0).

---

# PART A — Plan 2: Embedder

## Task A0: Sync poke-env fork to upstream for data currency [OPS]

**Files:** poke-env fork repo at `/home/cayman/Repositories/poke-env`.

- [ ] **Step 1: Preserve the fork's uncommitted edits**

```bash
cd /home/cayman/Repositories/poke-env
git status -sb
git stash push -u -m "wip before upstream sync"   # or commit them on a branch if they are intended changes
```
Acceptance: `git status` is clean.

- [ ] **Step 2: Fetch upstream and diff the gen9 data**

```bash
git fetch upstream
git diff --stat HEAD upstream/master -- src/poke_env/data/static/pokedex src/poke_env/data/static/moves
```
Acceptance: you can see whether the pokedex/move JSON differs. If there is **no** diff in the data files, skip to Step 5 (no merge needed) and restore the stash.

- [ ] **Step 3: Merge upstream (only if there is a data delta)**

```bash
git merge upstream/master
# resolve conflicts preserving the EFAI custom commits:
#   - ObservedPokemon restoration
#   - battle.observations (events-only Observation)
#   - DoubleBattle team-order sync on switch/swap
```
Acceptance: merge completes; the three EFAI customizations are present (`git log --oneline | grep -iE "observedpokemon|observations|team order"`).

- [ ] **Step 4: Restore the stashed edits**

```bash
git stash pop   # resolve any conflicts
```

- [ ] **Step 5: Re-run the FULL EFAI suite (poke-env bump = project-wide blast radius)**

```bash
cd /home/cayman/Repositories/EliteFurretAI
source ../venv/bin/activate && pytest unit_tests -q
```
Acceptance: no NEW failures vs the pre-merge baseline. If the merge introduced data that changed `embedder.embedding_size`, that is expected and handled by Task A4 re-processing; note it.

- [ ] **Step 6: Commit (poke-env repo) if merged**

```bash
cd /home/cayman/Repositories/poke-env
git log --oneline -3   # confirm merge commit present; push is the user's call
```

---

## Task A1: Mega Stone item vocab from the pokedex [TDD]

**Files:**
- Modify: `src/elitefurretai/etl/embedder.py` (add helper near the other `build_*` functions ~line 1772; union stones into the item vocab ~line 1771)
- Test: `unit_tests/etl/test_embedder.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_mega_stone_to_species_and_item_vocab():
    from elitefurretai.etl.embedder import (
        build_mega_stone_to_species,
        ITEM_TO_ID,
    )

    mapping = build_mega_stone_to_species(9)
    # Known stones map to their mega forme species keys
    assert mapping["venusaurite"] == "venusaurmega"
    assert mapping["charizarditex"] == "charizardmegax"
    assert mapping["charizarditey"] == "charizardmegay"
    # Rayquaza-Mega has no stone (requiredMove), so no rayquaza entry
    assert all(v != "rayquazamega" for v in mapping.values())
    # Stones are present in the item id space (nonzero ids)
    assert ITEM_TO_ID.get("venusaurite", 0) > 0
    assert ITEM_TO_ID.get("charizarditex", 0) > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py::test_mega_stone_to_species_and_item_vocab -v`
Expected: FAIL — `build_mega_stone_to_species` does not exist / stones not in `ITEM_TO_ID`.

- [ ] **Step 3: Implement the helper and union stones into the item vocab**

In `src/elitefurretai/etl/embedder.py`, add near the other builders (the file already imports `to_id_str` and `GenData`):

```python
def build_mega_stone_to_species(gen: int) -> Dict[str, str]:
    """Map Mega Stone item id -> mega-forme species key, from the gen pokedex.

    Derived from each mega forme's ``requiredItem`` field. The mega forme entry
    also carries the prospective baseStats / types / abilities the embedder reads.
    Rayquaza-Mega uses ``requiredMove`` (Dragon Ascent), so it has no stone here.
    """
    pokedex = GenData.from_gen(gen).pokedex
    mapping: Dict[str, str] = {}
    for species_key, entry in pokedex.items():
        forme = str(entry.get("forme", ""))
        required_item = entry.get("requiredItem")
        if forme.startswith("Mega") and required_item:
            mapping[to_id_str(str(required_item))] = species_key
    return mapping
```

Then change the item-vocab construction (currently `ITEM_TO_ID = {item: i + 1 for i, item in enumerate(sorted(TRACKED_ITEMS))}` at ~line 1771) to union in the Mega Stones:

```python
MEGA_STONE_TO_SPECIES = build_mega_stone_to_species(DEFAULT_GEN)
_ALL_TRACKED_ITEMS = TRACKED_ITEMS | set(MEGA_STONE_TO_SPECIES.keys())
ITEM_TO_ID = {item: i + 1 for i, item in enumerate(sorted(_ALL_TRACKED_ITEMS))}
```

Note `DEFAULT_GEN` is defined at line ~1810 *after* `ITEM_TO_ID` today. Move the `DEFAULT_GEN = 9` assignment above the `MEGA_STONE_TO_SPECIES` line (it is a plain constant), or inline `9`. Keep `NUM_ITEMS = len(_ALL_TRACKED_ITEMS) + 1`.

- [ ] **Step 4: Run to verify it passes**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py::test_mega_stone_to_species_and_item_vocab -v`
Expected: PASS.

- [ ] **Step 5: Run the embedder suite for regressions**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py unit_tests/etl/test_embedder_improvements.py -q`
Expected: all PASS (item-id values shift, but tests should key off presence/relative ids, not absolute; if any asserts an absolute item id, update it to match the new sorted vocab).

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/etl/embedder.py unit_tests/etl/test_embedder.py
git commit -m "feat(embedder): add Mega Stones to item vocab from pokedex requiredItem

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task A2: Prospective mega-form feature block [TDD]

**Files:**
- Modify: `src/elitefurretai/etl/embedder.py` — add `self._pokedex` in `__init__` (~after line 91 where `_knowledge["Pokemon"]` is set); add `_prospective_mega_entry` method; add the block to `generate_pokemon_features` (~after the TERA_TYPE block ending ~line 980) and `generate_opponent_pokemon_features` (~after its TERA_TYPE block ~line 1160).
- Test: `unit_tests/etl/test_embedder.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_prospective_mega_form_features_own_mon():
    from poke_env.battle.pokemon import Pokemon
    from elitefurretai.etl.embedder import Embedder

    embedder = Embedder()  # gen 9 default

    # A Venusaur holding Venusaurite: prospective mega is Grass/Poison, Thick Fat,
    # mega base stats {hp80, atk100, def123, spa122, spd120, spe80}.
    mon = Pokemon(gen=9, species="venusaur")
    mon.item = "venusaurite"

    feats = embedder.generate_pokemon_features(mon, "MON:0:", battle=None, dummy=True) \
        if False else None
    # Use the public per-mon path the embedder exposes; if generate_pokemon_features
    # requires a battle, build features via the documented helper used in
    # test_generate_pokemon_features (mirror that test's call exactly).
```

Because `generate_pokemon_features` needs the surrounding battle/prefix wiring, mirror the existing `test_generate_pokemon_features` setup in this same file for the call convention. The assertions to add once features are obtained for a Venusaur+Venusaurite mon (prefix `P`):

```python
    assert feats[P + "MEGA_STAT:def"] == 123.0
    assert feats[P + "MEGA_STAT:spa"] == 122.0
    # Multi-hot typing: Grass AND Poison set, others 0
    assert feats[P + "MEGA_TYPE:GRASS"] == 1
    assert feats[P + "MEGA_TYPE:POISON"] == 1
    assert feats[P + "MEGA_TYPE:FIRE"] == 0
    # mega ability id resolves Thick Fat (nonzero, known ability)
    assert feats[P + "mega_ability_id"] > 0

def test_no_prospective_mega_without_stone():
    from poke_env.battle.pokemon import Pokemon
    from elitefurretai.etl.embedder import Embedder
    embedder = Embedder()
    mon = Pokemon(gen=9, species="venusaur")  # no item
    # mirror the call convention; with no recognized stone the block is the
    # "unknown" sentinel (-1) for stats and ability and types
    # assert feats[P + "MEGA_STAT:def"] == -1
    # assert feats[P + "MEGA_TYPE:GRASS"] == -1
    # assert feats[P + "mega_ability_id"] == -1
```

(The implementer must wire these two tests to the exact `generate_pokemon_features` call signature used by `test_generate_pokemon_features` at [test_embedder.py:129](../../unit_tests/etl/test_embedder.py); copy that setup so the call matches.)

- [ ] **Step 2: Run to verify it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py -k "prospective_mega" -v`
Expected: FAIL — `MEGA_STAT:`/`MEGA_TYPE:`/`mega_ability_id` keys absent.

- [ ] **Step 3: Implement**

In `__init__`, add a pokedex handle (after `self._knowledge["Pokemon"] = ...`):

```python
        self._pokedex = GenData.from_gen(self._gen).pokedex
```

Add the helper (method on `Embedder`):

```python
    def _prospective_mega_entry(self, mon) -> Optional[dict]:
        """The pokedex entry for the mega forme this mon could evolve into, or None.

        Valid only when the mon holds a recognized Mega Stone whose forme's
        baseSpecies matches the mon's (base) species. Visible before mega-evolving,
        mirroring how TERA_TYPE exposes the prospective post-tera type.
        """
        if mon is None or not mon.item:
            return None
        species_key = MEGA_STONE_TO_SPECIES.get(to_id_str(mon.item))
        if species_key is None:
            return None
        entry = self._pokedex.get(species_key)
        if entry and to_id_str(str(entry.get("baseSpecies", ""))) == to_id_str(mon.species):
            return entry
        return None
```

Add this block in BOTH `generate_pokemon_features` and `generate_opponent_pokemon_features`, immediately after their existing `TERA_TYPE` loop:

```python
        # Prospective mega form (mirrors TERA_TYPE; visible before mega-evolving).
        mega_entry = self._prospective_mega_entry(mon)
        for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
            emb[prefix + "MEGA_STAT:" + stat] = (
                float(mega_entry["baseStats"][stat]) if mega_entry else -1
            )
        mega_types = (
            {str(t).lower() for t in mega_entry.get("types", [])} if mega_entry else set()
        )
        for ptype in self._knowledge["PokemonType"]:
            if ptype in [PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR]:
                continue
            emb[prefix + "MEGA_TYPE:" + ptype.name] = (
                int(ptype.name.lower() in mega_types) if mega_entry else -1
            )
        if mega_entry:
            mega_ability = to_id_str(str(mega_entry.get("abilities", {}).get("0", "")))
            emb[prefix + "mega_ability_id"] = self._ability_to_id.get(mega_ability, 0)
        else:
            emb[prefix + "mega_ability_id"] = -1
```

- [ ] **Step 4: Run to verify it passes**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py -k "prospective_mega" -v`
Expected: PASS.

- [ ] **Step 5: Regression + dimension sanity**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py unit_tests/etl/test_embedder_improvements.py -q`
Expected: all PASS. Note: `embedding_size` increases (expected; Task A4 re-processes). If a test asserts a hard-coded `embedding_size`, update it to the new value.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/etl/embedder.py unit_tests/etl/test_embedder.py
git commit -m "feat(embedder): prospective mega-form feature block (multi-hot MEGA_TYPE)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task A3: Mega scalar flags (is_mega_evolved, can_mega/can_tera, gimmick-spent) [TDD + verification spike]

**Files:**
- Modify: `src/elitefurretai/etl/embedder.py` (per-mon `is_mega_evolved` near `is_terastallized`; `can_mega`/`can_tera`/gimmick-spent in `generate_battle_features`)
- Modify: `src/elitefurretai/etl/battle_iterator.py` (track `|-mega|` if no poke-env signal exists)
- Test: `unit_tests/etl/test_embedder.py`

- [ ] **Step 1: Verification spike — find the mega-evolved signal**

```bash
cd /home/cayman/Repositories/poke-env
grep -rni "is_mega\|'-mega'\|\"-mega\"\|mega" src/poke_env/battle/pokemon.py src/poke_env/battle/abstract_battle.py | head
```
Decide the signal:
- If `Pokemon` exposes a usable mega-evolved property/flag → use it directly.
- Otherwise: have `BattleIterator` (in `src/elitefurretai/etl/battle_iterator.py`) record the `|-mega|p1a: Name|Stone` protocol event into a per-mon set and expose `mon`-keyed lookup, mirroring how it already tracks other per-mon state.
Document the chosen signal in the task's commit message.

- [ ] **Step 2: Write the failing test (battle-level flags, which are deterministic from `battle`)**

```python
def test_battle_features_mega_tera_availability_flags():
    from unittest.mock import MagicMock
    from poke_env.battle.double_battle import DoubleBattle
    from elitefurretai.etl.embedder import Embedder

    embedder = Embedder()
    battle = MagicMock(spec=DoubleBattle)
    battle.can_mega_evolve = [True, False]
    battle.can_tera = [False, False]
    # populate the other attrs generate_battle_features reads by mirroring the
    # existing test_generate_battle_features setup in this file.

    feats = embedder.generate_battle_features(battle)
    assert feats["CAN_MEGA:0"] == 1
    assert feats["CAN_MEGA:1"] == 0
    assert feats["CAN_TERA:0"] == 0
```

(Mirror the existing battle-features test setup in this file for the other required `battle` attributes.)

- [ ] **Step 3: Run to verify it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py -k "mega_tera_availability" -v`
Expected: FAIL — keys absent.

- [ ] **Step 4: Implement**

Per-mon, next to `is_terastallized` in both per-mon methods:

```python
        emb[prefix + "is_mega_evolved"] = <signal from Step 1> if mon else -1
```

In `generate_battle_features`, add (using the live `battle` lists, guarding length):

```python
        can_mega = getattr(battle, "can_mega_evolve", None) or [False, False]
        can_tera = getattr(battle, "can_tera", None) or [False, False]
        for slot in (0, 1):
            emb["CAN_MEGA:" + str(slot)] = int(bool(can_mega[slot])) if slot < len(can_mega) else -1
            emb["CAN_TERA:" + str(slot)] = int(bool(can_tera[slot])) if slot < len(can_tera) else -1
        # Per-side once-per-battle gimmick spent: any of our/their mons terastallized or mega-evolved.
        emb["OUR_GIMMICK_SPENT"] = int(
            any(getattr(m, "is_terastallized", False) for m in battle.team.values())
            or <our-side mega-evolved check from Step 1>
        )
        emb["OPP_GIMMICK_SPENT"] = int(
            any(getattr(m, "is_terastallized", False) for m in battle.opponent_team.values())
            or <opp-side mega-evolved check from Step 1>
        )
```

- [ ] **Step 5: Run to verify it passes + regression**

Run: `source ../venv/bin/activate && pytest unit_tests/etl/test_embedder.py unit_tests/etl/test_embedder_improvements.py -q`
Expected: all PASS (update any hard-coded `embedding_size` assertion).

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/etl/embedder.py src/elitefurretai/etl/battle_iterator.py unit_tests/etl/test_embedder.py
git commit -m "feat(embedder): mega scalar flags — is_mega_evolved, can_mega/can_tera, gimmick-spent

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task A4: Re-process tera training tensors with the new embedder [OPS]

**Files:** none (runs the existing pipeline).

- [ ] **Step 1: Confirm the embedding dimension changed as expected**

```bash
source ../venv/bin/activate && python -c "from elitefurretai.etl.embedder import Embedder; print(Embedder().embedding_size)"
```
Acceptance: prints the new (larger) dimension; record it.

- [ ] **Step 2: Re-run processing on the existing valid-file list**

```bash
source ../venv/bin/activate && python src/elitefurretai/etl/process_training_data.py
```
(Use the same input file-list/config the project normally uses; see `ETL.md`.)
Acceptance: produces `.pt.zst` chunks whose per-state feature width equals the new `embedding_size`. Verify on one chunk:
```bash
source ../venv/bin/activate && python -c "
import torch, zstandard, io, glob
# load one processed chunk per the project's loader and print the feature dim
from elitefurretai.etl.battle_dataset import BattleDataset  # use the documented loader
print('inspect a sample feature width matches new embedding_size')
"
```

- [ ] **Step 3: Commit any processing-manifest/config changes (not the large tensors)**

```bash
git add -A -- '*.yaml' '*.json'   # only manifests/configs, never the .pt.zst blobs
git commit -m "chore(etl): re-process tera tensors for mega-aware featureset" || echo "nothing to commit"
```

---

## Task A5: BC re-finetune with vocab-aware warm-start [TDD for the transfer fn + OPS for the run]

**Files:**
- Modify: `src/elitefurretai/rl/learners.py` (or `src/elitefurretai/supervised/utils.py`) — add a `transfer_compatible_weights(old_state, new_model)` that copies overlapping parameter slices (entity-embedding rows by id-overlap; `input_proj` columns up to the old width) and leaves the rest at init.
- Test: `unit_tests/rl/` (new small test for the transfer fn)

- [ ] **Step 1: Write the failing test for the transfer function**

```python
def test_transfer_compatible_weights_copies_overlap_and_pads():
    import torch
    from torch import nn
    from elitefurretai.rl.learners import transfer_compatible_weights

    old = nn.Linear(4, 3)
    nn.init.constant_(old.weight, 1.0)
    new = nn.Linear(6, 3)          # input grew 4 -> 6
    nn.init.constant_(new.weight, 0.0)

    transfer_compatible_weights(old.state_dict(), new, key="weight")
    w = new.weight.detach()
    assert torch.all(w[:, :4] == 1.0)   # old columns copied
    assert torch.all(w[:, 4:] == 0.0)   # new columns left at init (zero)
```

- [ ] **Step 2: Run to verify it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl -k "transfer_compatible_weights" -v`
Expected: FAIL — function does not exist.

- [ ] **Step 3: Implement `transfer_compatible_weights`**

```python
def transfer_compatible_weights(old_state: dict, new_module, key: str) -> None:
    """Copy the overlapping leading slice of a 2D weight from old_state into
    new_module[key], leaving any grown rows/columns at their current init.

    Used to warm-start a model whose input projection (and entity-embedding
    tables) grew when mega features were added: existing learned weights are
    preserved; only the genuinely-new slices start fresh.
    """
    import torch

    new_param = dict(new_module.named_parameters())[key]
    old_w = old_state[key]
    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_w.shape, new_param.shape))
    with torch.no_grad():
        new_param[slices].copy_(old_w[slices])
```

- [ ] **Step 4: Run to verify it passes**

Run: `source ../venv/bin/activate && pytest unit_tests/rl -k "transfer_compatible_weights" -v`
Expected: PASS.

- [ ] **Step 5 [OPS]: Warm-start from cool-bee-85 and re-finetune BC**

Wire the BC finetune entry point to: build the model at the new `embedding_size`, load `data/models/supervised/cool-bee-85-finetune_best.pt` with `transfer_compatible_weights` applied to `input_proj.weight` and each entity-embedding table (rows copied for ids present in both old and new vocab — item ids shift because Mega Stones were inserted into the sorted vocab, so map by item *name*, not index), then finetune on the re-processed tera data.
Acceptance: BC validation accuracy on tera data is within noise of cool-bee-85's (new features are constant on tera data, so behavior should match). Save as `cool-bee-85-mega-finetune_best.pt`.

- [ ] **Step 6: Update SUPERVISED.md + commit the checkpoint pointer (not weights)**

```bash
git add src/elitefurretai/supervised/SUPERVISED.md src/elitefurretai/rl/learners.py unit_tests/rl/*.py
git commit -m "feat(rl): vocab-aware warm-start; re-finetune BC for mega featureset

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# PART B — Plan 3: Opponent baselines

## Task B1: Mega-aware enumeration in heuristic opponents [TDD]

**Files:**
- Modify: `src/elitefurretai/agents/max_damage_player.py` (its move-choice path)
- Modify: the simple-heuristic path used by `simple_heuristic_baseline` (poke-env `SimpleHeuristicsPlayer` wrapper in `opponents.py`, or a thin subclass if the move-emission is wrapped there)
- Test: `unit_tests/rl/test_opponent_pool.py` (append)

- [ ] **Step 1: Verification — how does max_damage currently emit tera?**

```bash
grep -ni "tera\|mega\|create_order\|can_tera\|can_mega" src/elitefurretai/agents/max_damage_player.py
```
Mirror the existing tera path for mega: where it would emit `terastallize=True` (or never gimmicks), allow emitting `mega=True` when `battle.can_mega_evolve[slot]`.

- [ ] **Step 2: Write the failing test**

```python
def test_max_damage_emits_mega_when_available():
    # Build a DoubleBattle (or MagicMock spec) where can_mega_evolve=[True, False]
    # and a clear best-damage move exists on slot 0; assert the chosen order for
    # slot 0 has mega=True. Mirror the existing max_damage test setup in this file.
    ...
```

(Wire to the existing max_damage test harness in the repo; if none exists, build a minimal `DoubleBattle` mock mirroring `test_fast_action_mask.py`'s approach and assert on the emitted `BattleOrder.mega`.)

- [ ] **Step 3: Run to verify it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_opponent_pool.py -k "mega" -v`
Expected: FAIL.

- [ ] **Step 4: Implement mega emission**

In the max_damage choose path, when a non-fainted active slot has `battle.can_mega_evolve[slot]` truthy, emit the chosen move with `mega=True` (mega is strictly beneficial for a max-damage heuristic since the mega form has ≥ base offensive stats). For `simple_heuristic`, do the same gating in its move-emission wrapper. Keep the once-per-turn / one-slot constraint (only mega one slot), consistent with the mask.

- [ ] **Step 5: Run to verify it passes + regression**

Run: `source ../venv/bin/activate && pytest unit_tests/rl/test_opponent_pool.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/agents/max_damage_player.py src/elitefurretai/rl/opponents.py unit_tests/rl/test_opponent_pool.py
git commit -m "feat(opponents): mega-aware enumeration for max_damage and simple_heuristic

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# PART C — Plan 4: Curriculum + engine validation

## Task C1: Add regMA to the curriculum with a tera:mega ratio knob [TDD]

**Files:**
- Modify: `src/elitefurretai/rl/config.py` (add the ratio field with a default)
- Modify: the active curriculum yaml (`src/elitefurretai/rl/configs/may26.yaml`)
- Test: `unit_tests/rl/` (config-loading test)

- [ ] **Step 1: Verification — how are formats sampled today?**

```bash
grep -rni "format\|regh\|sample\|curriculum\|weight" src/elitefurretai/rl/config.py | head -30
```
Identify the format-sampling structure so the ratio knob slots into the existing schema (do not invent a parallel mechanism).

- [ ] **Step 2: Write the failing test**

```python
def test_curriculum_includes_regma_with_ratio_default():
    from elitefurretai.rl.config import <ConfigType>   # the curriculum/agent-axis config
    cfg = <ConfigType>.from_yaml("src/elitefurretai/rl/configs/may26.yaml")
    formats = <accessor for the sampled formats>
    assert "gen9championsvgc2026regma" in formats
    # ratio knob exists with a sensible default (tuned at runtime)
    assert hasattr(cfg, "tera_mega_ratio") or "<knob path>" in cfg.<...>
```

(Bind to the real config type/accessor found in Step 1.)

- [ ] **Step 3: Run to verify it fails**

Run: `source ../venv/bin/activate && pytest unit_tests/rl -k "regma_with_ratio" -v`
Expected: FAIL.

- [ ] **Step 4: Implement**

Add regMA to the format set in `may26.yaml` and a `tera_mega_ratio` (or format-weight) knob in `config.py` with a documented default (start ~proportional / balanced; tuned at runtime). Wire the knob into the existing format-sampling code path (found in Step 1) — do not add a second sampler.

- [ ] **Step 5: Run to verify it passes + config regression**

Run: `source ../venv/bin/activate && pytest unit_tests/rl -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elitefurretai/rl/config.py src/elitefurretai/rl/configs/may26.yaml unit_tests/rl/*.py
git commit -m "feat(rl): add regMA to curriculum with tera:mega ratio knob

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task C2: Engine end-to-end validation on showdown_websocket [OPS]

**Files:** none (live-server validation); fixes (if any) land in `engine/` or `masking.py`/`encoder.py` as separate commits.

- [ ] **Step 1: Run a small regMA self-play / heuristic battle set on showdown_websocket**

Use the project's standard short run targeting `gen9championsvgc2026regma` with regMA teams from `data/teams/gen9championsvgc2026regma/`, the new BC checkpoint, and the mega-aware opponents (Part B).
Acceptance: battles complete with **zero invalid-choice errors** over the run.

- [ ] **Step 2: Sweep for new invalid-choice families**

Reuse `src/elitefurretai/engine/analyze/showdown_invalid_choice_diagnostics.py` to surface any mega-specific rejections (mega + targeting/redirection, mega + Sky Drop, etc.).
Acceptance: diagnostics report no unhandled mega-related rejection classes; any found are root-caused and fixed (mask/encoder/validator) with their own TDD commits.

- [ ] **Step 3: Record results in a planning doc**

Add a dated entry to `planning/stage2/2026-05-27-17-13-mega-format-etl-rl-support-design.md` Updates with the validation outcome and any residual issues.

---

## Self-Review

**Spec coverage (against the combined design):**
- 2.1 data currency → Task A0 (fork merge-from-upstream). ✓
- 2.2 curated vocab (Mega Stones, effects) → Task A1 (stones from pokedex). Effects: add to `TRACKED_EFFECTS` *if* the verification spike in C2/A0 surfaces new regMA volatile statuses — folded into A0/C2 findings rather than a speculative task (YAGNI: no known new effects to enumerate yet). ✓ (flagged)
- 2.3 prospective mega-form block (multi-hot MEGA_TYPE) → Task A2. ✓
- 2.4 scalar flags → Task A3. ✓
- 2.5 re-process + re-finetune with warm-start → Tasks A4, A5. ✓
- Plan 3 opponents → Task B1. ✓
- Plan 4 curriculum + engine → Tasks C1, C2. ✓

**Placeholder scan:** TDD tasks A1, A2(partial), A3(battle flags), A5(transfer fn) carry complete code. A2/A3/B1/C1 test *bodies* defer the fixture-wiring to "mirror the existing test in this file" because the exact call setup lives in the current test module and must match it verbatim — this is a deliberate instruction to copy a known local pattern, not an unfilled TODO. The OPS tasks (A0, A4, A5-run, C2) give exact commands + acceptance criteria; they are not unit-testable.

**Type/name consistency:** `build_mega_stone_to_species`, `MEGA_STONE_TO_SPECIES`, `_prospective_mega_entry`, `MEGA_STAT:`, `MEGA_TYPE:`, `mega_ability_id`, `is_mega_evolved`, `CAN_MEGA:`, `CAN_TERA:`, `OUR_/OPP_GIMMICK_SPENT`, `transfer_compatible_weights` used consistently. Feature naming follows the existing `PREFIX + "KEY"` convention.

**Known dependencies between tasks:** A2/A3 depend on A1 (`MEGA_STONE_TO_SPECIES`). A4 depends on A1–A3 (final featureset). A5 depends on A4. C2 depends on A5 + B1 + C1. A3's per-mon flag and gimmick-spent depend on the Step-1 verification spike outcome.

## Open items (carried from design)
- tera:mega ratio default value (set a sensible default in C1; tune at runtime).
- regMA graduation bar (parked; decide after convergence).
- New regMA volatile statuses/effects for `TRACKED_EFFECTS` (add only if A0/C2 surface any).
