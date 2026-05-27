# Mega-Evolution Format: Cross-Format Pipeline Support (Design)

## Context

A new VGC format (`gen9championsvgc2026regma`, "regMA") uses **mega-evolution** in place
of terastallization. The roster of species and moves is largely shared with existing tera
formats. The long-term goal is a single agent that plays **across** tera formats and the
mega format.

Constraints established up front:
- **No supervised battle data** exists for the mega format, so format-specific behavior
  cloning (BC) for mega is impossible. BC remains available on tera formats.
- Cross-format scope is **tera + mega only**. Gen-8 dynamax is explicitly out of scope, so
  the gimmick abstraction does not need to model multi-turn, move-remapping gimmicks.

Current pipeline grounding (verified against source, 2026-05-27):
- `gen9championsvgc2026regma` is already in `TRACKED_FORMATS`
  ([embedder.py:1711](../../src/elitefurretai/etl/embedder.py)) and has scraped teams under
  `data/teams/gen9championsvgc2026regma/`.
- poke-env + the order validator already understand mega: `order.mega`,
  `battle.can_mega_evolve`, and the "can't mega two mons" constraint all exist
  ([battle_order_validator.py:56-281](../../src/elitefurretai/etl/battle_order_validator.py)).

## Before State

The gimmick mechanic is wired for tera only in the learning-facing layers:

- **Action encoding** ([encoder.py:31-72](../../src/elitefurretai/etl/encoder.py)):
  `INT_TO_ORDER` hardcodes indices 0–39; within each 10-wide move group, offsets 5–9 mean
  "terastallize." Mega is absent from the action enum.
- **Masking** ([masking.py:212-517](../../src/elitefurretai/rl/masking.py)): tera offsets
  and the "no double-tera per turn" pair constraint are hardcoded; no mega awareness.
- **State features** ([embedder.py](../../src/elitefurretai/etl/embedder.py)): per-mon
  `is_terastallized` + `TERA_TYPE` one-hot, plus a `FORMAT:` one-hot. No `can_tera`
  availability feature, and nothing representing a mega form or a generic "gimmick spent"
  flag.
- **Graduation** ([opponents.py:145-169](../../src/elitefurretai/rl/opponents.py)): Stage II
  bar is ≥60% simultaneously vs `vgc_bench`, `max_damage`, `bc_player`,
  `simple_heuristic_baseline`.

## Problem

To train one agent across tera and mega formats, the action space, mask, and state features
must represent mega-evolution, and they must do so in a way that lets knowledge learned on
tera formats (where BC data exists) transfer into the mega format (where it does not).
Mega-evolution is mechanically analogous to tera — a once-per-team, once-per-turn modifier
attached to a move — but it transforms the Pokémon more deeply (species, typing, ability,
**and base stats**, gated by holding a Mega Stone), and it is **mutually exclusive** with
tera within any given battle/format.

The graduation criterion also breaks on the mega format: `bc_player` cannot exist there.

## Solution

### Keystone decision (locked): unified gimmick slot + shared format-conditioned policy

- **Action space stays 40.** The existing tera offsets (5–9 within each move group) become a
  generic **"gimmick"** offset. `MDBO.to_double_battle_order` emits `terastallize=True` vs
  `mega=True` based on battle legality (`battle.can_mega_evolve` / `can_tera`), which the
  validator already supports. Because the two gimmicks never co-occur in one battle, this is
  unambiguous at emission time.
- **One shared policy**, conditioned on the existing `FORMAT:` one-hot. This is the only
  design that lets tera-format competence (team preview, switching, targeting, damage trades,
  and the general "a once-per-battle resource has opportunity cost" prior) transfer into the
  mega format, where there is no supervised signal.

Rationale for not adding separate mega offsets: since tera and mega are mutually exclusive
per battle, separate offsets would permanently mask-dead half the gimmick logits in every
battle, widen the policy head for no semantic gain, and give the mega half zero BC gradient.

### Mega-aware state features (orthogonal work)

The state embedding must represent mega regardless of the action encoding:
1. A generic **"gimmick spent this battle"** flag (shared semantic across tera/mega) so the
   shared policy can reason about resource use uniformly.
2. A **mega-availability** signal (analogous to the missing `can_tera` — worth adding both
   for symmetry).
3. The **mega form's altered stats / typing / ability**, gated on holding the Mega Stone.
   Tera's type-only change does not cover this.

### Masking

Generalize the hardcoded tera-offset logic in `masking.py` into one gimmick rule
parameterized by which gimmick is legal in the current battle (`can_tera` vs
`can_mega_evolve`), preserving the once-per-turn / can't-gimmick-while-switching constraints.
This is the same invalid-choice bug class recently fixed; the mask must match Showdown's
mega rules exactly.

### Learning signal

- **BC** runs on tera formats only; ~95% of play is format-agnostic and bootstraps general
  competence. Under the unified slot, the BC-learned "when to spend the gimmick" prior
  partially transfers to mega timing.
- **r-NaD anchor**: no change needed. `PortfolioRNaDLearner` regularizes (min-KL) against a
  rolling portfolio of **snapshots of the policy itself**, not a frozen BC policy
  ([learners.py:33-36](../../src/elitefurretai/rl/learners.py)). On the mega format those
  snapshots are mega-playing policies, so the anchor is well-defined and co-evolves with mega
  learning. BC is only the backbone initialization, not the anchor. Initial clumsy mega
  snapshots are absorbed transiently by min-KL across the portfolio; no format-masking.
- **Reward / self-play**: win-loss reward is format-agnostic; confirm nothing in the
  trajectory/reward path assumes tera.

### Data, opponents, evaluation

- **Teams**: regMA teams must carry legal Mega Stones; `scrape_pastes.py` already enforces
  level 50. Validate stone↔species legality.
- **Curriculum**: introduce mega from day one and bias broadly for exposure (consistent with
  the curriculum-design preference); choose a deliberate tera:mega sampling mix.
- **Opponents**: `max_damage` and `simple_heuristic` need mega-aware action enumeration;
  `vgc_bench` may not know regMA. `bc_player` cannot be a meaningful mega baseline.
- **Graduation bar (OPEN)**: the four-baseline criterion can't be reused on regMA because
  `bc_player` is structurally absent. Deferred — settle once mega RL convergence is observed.

### Engine & inference

- `showdown_websocket` must run regMA cleanly; mega introduces a fresh family of potential
  invalid-choice edge cases (mega + targeting/redirection).
- `speed_inference` / `item_inference` assume tera-era mechanics; a Mega Stone is a revealing
  held item and mega changes base speed/ability mid-battle. Update opponent-modeling inference
  if it is used in the relevant path.

## Reasoning

The mutual-exclusivity of tera and mega per battle is the fact that makes the unified slot
clearly correct rather than merely convenient: there is no in-battle ambiguity to resolve,
and the alternative wastes head capacity. The shared-trunk + FORMAT-conditioning bet is the
standard way to share a representation while letting a conditioning feature carve out
format-specific behavior — and it is essentially forced here, because mega timing is learned
from RL cold-start regardless of the encoding, so starting from a transferred "gimmicks have
opportunity cost" prior strictly beats a uniform start. The main residual risk — a
tera-flavored *reactive* timing prior bleeding into mega's more *proactive* optimal timing —
is exactly what FORMAT conditioning plus RL on regMA corrects; it slows convergence, it does
not cap it.

## Planned Next Steps

1. **Action layer**: generalize `INT_TO_ORDER` semantics (gimmick offset) and
   `MDBO.to_double_battle_order` to emit `mega`/`terastallize` by battle legality.
2. **Masking**: parameterize the gimmick offset + pair constraint by legal gimmick.
3. **State features**: add gimmick-spent flag, mega-availability (and `can_tera` for
   symmetry), and mega-form stat/type/ability representation to the embedder.
4. **Opponents**: make `max_damage` / `simple_heuristic` enumerate mega actions; confirm
   `vgc_bench` regMA support.
5. **Curriculum**: add regMA to the sampling mix from the start; pick tera:mega ratio.
6. **Engine**: validate regMA runs end-to-end on `showdown_websocket`; sweep for new
   invalid-choice families.
7. **Inference**: update `speed_inference` / `item_inference` for mega forms if in path.
8. **Graduation (open)**: define the regMA bar after observing convergence.

(r-NaD requires no change — see Solution; the portfolio-of-self-snapshots anchor handles a
no-BC format natively.)

## Updates

- 2026-05-27 16:15 — Initial design. Locked: unified gimmick slot + shared
  format-conditioned policy; scope tera + mega only (dynamax out). Open: regMA graduation
  criterion (bc_player structurally absent).
- 2026-05-27 16:20 — Dropped the proposed r-NaD anchor format-masking. The anchor is a
  rolling portfolio of self-snapshots (not a frozen BC policy), so it co-evolves with mega
  learning and needs no change. Corrects an earlier wrong assumption that the anchor was the
  BC policy.
- 2026-05-27 16:45 — Plan 1 (action layer) shipped on branch `mega-format-support`
  (commits c2f25cd → 64e6b98): the gimmick offset now represents mega as well as tera —
  mask reads `canMegaEvo` (`masking.py`), encoder emits `mega` vs `terastallize` from
  `battle.can_mega_evolve` (`encoder.py`). Integer action space, BC parsing, and policy head
  unchanged. Next: Plan 2 (embedder mega-form features).
