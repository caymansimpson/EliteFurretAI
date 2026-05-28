# Mega-Format ETL→RL Support: Embedder + Opponents + Curriculum (Design)

## Context

Plan 1 (the mega action layer) shipped on `main`: the once-per-battle "gimmick"
action offset now represents mega-evolution as well as terastallization, the mask reads
`canMegaEvo`, and the encoder emits `mega` vs `terastallize` from `battle.can_mega_evolve`.
See [`2026-05-27-16-15-mega-format-cross-format-support-design.md`](2026-05-27-16-15-mega-format-cross-format-support-design.md)
and [`2026-05-27-16-30-mega-action-layer-implementation-plan.md`](2026-05-27-16-30-mega-action-layer-implementation-plan.md).

This design covers the next three subsystems, to be combined into one implementation effort
(the user requested combining Plans 2–4):
- **Plan 2** — Embedder: vocab, data currency, and prospective mega-form features.
- **Plan 3** — Opponent baselines: mega-aware action enumeration.
- **Plan 4** — Curriculum + engine validation.

Constraints carried forward: no supervised data for the mega format (`gen9championsvgc2026regma`,
"regMA"), so BC stays on tera formats; scope is tera + mega only (dynamax out); the goal is one
agent playing across tera and mega formats.

## Before State

- **Embedder** ([`embedder.py`](../../src/elitefurretai/etl/embedder.py)): per-mon `STAT:*`,
  multi-hot `TYPE:*` ([line 968](../../src/elitefurretai/etl/embedder.py): a bit for
  `type_1 == ptype or type_2 == ptype`), `ability_id`, `item_id`, `species_id`,
  `is_terastallized`, single-hot `TERA_TYPE:*`, and a `FORMAT:` one-hot (regMA already in
  `TRACKED_FORMATS`). No mega-form, mega-availability, or gimmick-spent features.
- **Vocab builders**: `build_species_to_id/ability/move(gen)` enumerate the entire gen9
  `GenData` dex/movedex (gen9 dex already contains all 101 mega forms). `TRACKED_ITEMS` and
  `TRACKED_EFFECTS` are hand-curated; `TRACKED_ITEMS` contains no Mega Stones.
- **Model input dim** = `embedder.embedding_size` ([model_archs.py:1011](../../src/elitefurretai/supervised/model_archs.py));
  the entity-id encoder maps features by position via `group_embedding_sizes`. Any feature
  change alters the input dim and the cached `.pt.zst` tensors, and makes existing BC
  checkpoints (cool-bee-85) dimension-incompatible at the input projection and at any grown
  entity-embedding table.
- **poke-env fork** (`/home/cayman/Repositories/poke-env`): `origin` =
  `caymansimpson/poke-env`, `upstream` = `hsahovic/poke-env`. Carries EFAI customizations
  (ObservedPokemon restoration, `battle.observations`, DoubleBattle team-order sync). Gen9
  data refreshes arrive from upstream as "JSON data update from smogon" commits. The fork has
  uncommitted edits in `player.py` / `ps_client.py`.
- **Opponents** ([`opponents.py`](../../src/elitefurretai/rl/opponents.py)): `max_damage`,
  `simple_heuristic`, `vgc_bench`, `bc_player`. The heuristic baselines enumerate tera actions
  but not mega.
- **Mega-form representation in poke-env**: `Pokemon.mega_evolve(stone)` calls
  `_update_from_pokedex(mega_species, store_species=False)`, so after mega the mon's
  `STAT:*`, `TYPE:*`, and ability (via `forme_change_ability`) update live, while `species`
  stays the base form. There is no direct `is_mega` property (only `is_dynamaxed`).

## Problem

To train and evaluate one agent on the mega format, the embedder must represent mega
(including the *prospective* mega form so the policy can make the anticipatory mega decision),
the entity vocabularies must cover regMA's content (Mega Stones especially), the heuristic
opponents must play mega, and the RL curriculum and engine must run regMA. Because the feature
set changes, the cached training tensors and the BC checkpoint must be migrated.

## Solution

### Plan 2 — Embedder

**2.1 Data currency via poke-env fork sync (not a bespoke gap-checker).**
Species/abilities/moves auto-populate from gen9 `GenData`, so currency is a poke-env-data
question, resolved by syncing the fork to upstream:
1. Commit (or stash) the fork's uncommitted `player.py` / `ps_client.py` edits.
2. `git fetch upstream`; diff the gen9 data JSON (pokedex / moves / abilities) between the
   fork's current `master` and `upstream/master`. If there is no delta containing regMA
   content, skip the rest of 2.1.
3. If there is a delta: **merge** `upstream/master` into the fork (merge, not rebase —
   preserves the EFAI custom commits without replaying them onto a moving upstream and does
   not rewrite history pushed to `origin`), resolving conflicts while preserving the EFAI
   customizations.
4. Re-run the **full** EFAI test suite — a poke-env bump has project-wide blast radius.

**2.2 Curated-vocab additions.** Add Mega Stones (and any new regMA held items) to
`TRACKED_ITEMS`; add any new volatile statuses / effects introduced by regMA moves to
`TRACKED_EFFECTS`. These are the only hand-maintained vocabs.

**2.3 Prospective mega-form feature block (per mon).** Mirroring `TERA_TYPE` but for the full
mega form: for any mon holding a recognized Mega Stone, compute the prospective mega species
(poke-env's rule: `base + "mega"`, plus the stone's trailing `x`/`y` for split megas) and
encode from that pokedex entry:
- `MEGA_STAT:{hp,atk,def,spa,spd,spe}` — the prospective mega base/derived stats.
- `MEGA_TYPE:*` — **multi-hot** over the prospective form's `type_1` and `type_2`, mirroring
  the existing `TYPE:` block (NOT single-hot like `TERA_TYPE`, because mega keeps and often
  changes dual typing).
- `mega_ability_id` — the prospective mega ability (entity-id index).

This block is visible *before* and independent of actually mega-evolving, so the policy can
reason about the mega decision anticipatorily. On tera formats no mon holds a Mega Stone, so
the block is constant (-1 / unknown) and harmless. For opponent mons it is unknown until the
Mega Stone item is revealed (handled like other partial-information features). Encoded for the
same set of mon slots as the existing per-mon features (own team + observed opponents).

**2.4 Scalar flags.**
- `is_mega_evolved` per mon (mirrors `is_terastallized`). poke-env has no direct property; the
  implementation will pin a reliable signal (e.g. tracking the `|-mega|` battle event or a
  forme-change indicator) via a small verification before relying on it.
- `can_mega` / `can_tera` availability (from `battle.can_mega_evolve` / `battle.can_tera`).
- per-side "gimmick spent this battle" flag.

**2.5 Re-process + re-finetune BC.** Re-process the tera training tensors with the new
embedder (new dim). Re-finetune BC from cool-bee-85 with **vocab-aware warm-start**: copy
preserved entity-embedding rows (existing items/species/abilities) and the existing input_proj
columns, and zero/random-init only the genuinely new rows (e.g. Mega Stone item rows) and new
input columns (the mega feature block + scalars). On tera data the new features are constant,
so this preserves tera behavior; the new parameters are trained by mega RL (and warm-started by
the re-finetune). Produces a new mega-aware BC checkpoint.

### Plan 3 — Opponent baselines (independent; depends only on the shipped action layer)

Make `max_damage` and `simple_heuristic` enumerate mega actions — emit a mega order when
`battle.can_mega_evolve` is set for the slot, mirroring their existing tera handling. Confirm
or decide `vgc_bench` regMA support (it may not know the format; if not, it is simply absent
from the regMA baseline set, like `bc_player`).

### Plan 4 — Curriculum + engine validation

- Add regMA to the RL curriculum's format sampling. Expose the tera:mega ratio as a **config
  knob with a sensible default** (tuned at runtime, per the deferred decision).
- Validate regMA end-to-end on `showdown_websocket`; sweep for new invalid-choice families
  introduced by mega (e.g. mega + targeting/redirection interactions), mirroring the earlier
  Commander/Uproar/force-switch fixes.
- The regMA graduation bar stays parked (bc_player structurally absent; decide after observing
  convergence).

### Sequencing

Plan 2 first — it changes the feature/data/checkpoint contract everything else depends on.
Plan 3 is independent and can land any time after the action layer (already shipped). Plan 4
last — it needs Plan 2's checkpoint + data and Plan 3's opponents to train and evaluate.

## Reasoning

The prospective mega-form block is the crux: encoding the mega form only *after* mega-evolution
would force the policy to learn a deterministic (base species, stone) → (mega stats/typing/
ability) lookup purely from sparse RL reward with no BC signal, when that mapping is fully known
from the pokedex and the mega decision is inherently anticipatory. The existing design already
encodes the prospective post-tera type (`TERA_TYPE`) for exactly this reason; mega gets the
symmetric, fuller treatment because it changes more than type. `MEGA_TYPE` must be multi-hot
because mega forms retain dual typing.

Re-process + re-finetune (over input-layer surgery alone) is required because the entity
vocabularies grow (Mega Stones at minimum), not just appended scalars — the embedding tables
themselves change shape. Syncing the poke-env fork to upstream is the right way to get data
currency because the vocab builders derive entirely from `GenData`; merge (not rebase)
preserves the EFAI custom commits and published history. Win/loss reward and r-NaD are unchanged
(the portfolio-of-self-snapshots anchor handles a no-BC format natively; established earlier).

## Planned Next Steps

Write the combined implementation plan (`writing-plans`), structured as three task groups
(Plan 2 → embedder/vocab/data/checkpoint; Plan 3 → opponents; Plan 4 → curriculum/engine) with
Plan 2 sequenced first. Open items to settle during/after: tera:mega ratio default value, and
the regMA graduation bar.

## Updates

- 2026-05-27 17:13 — Initial design for combined Plans 2–4. Locked: prospective mega-form
  feature block (multi-hot `MEGA_TYPE`); data currency via poke-env fork merge-from-upstream
  (not a bespoke checker); re-process + re-finetune BC with vocab-aware warm-start; tera:mega
  ratio as a tuned config knob. Open: ratio default, regMA graduation bar.
