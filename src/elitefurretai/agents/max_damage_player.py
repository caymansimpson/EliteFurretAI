"""MaxDamagePlayer — heuristic baseline that picks the highest-damage action.

Non-learning Player used as a curriculum opponent and as a baseline in the
Stage II graduation criterion. Wraps poke_env.calc.calculate_damage.

See agents/AGENTS.md for usage.

Moved here from ``rl/players.py`` on 2026-05-19.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.calc import calculate_damage
from poke_env.data import GenData
from poke_env.player import BattleOrder, DoubleBattleOrder, Player
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.stats import compute_raw_stats

from elitefurretai.rl.masking import get_valid_targets, slot_is_commanding

logger = logging.getLogger("MaxDamagePlayer")


class MaxDamagePlayer(Player):
    """A non-learning, heuristic-only opponent for the curriculum.

    Why we have this
    ----------------
    Pure self-play is unstable: the agent can fall into degenerate cycles where
    it only learns to beat its current self. A diverse curriculum of opponents
    helps. MaxDamage is one of the simplest useful baselines:
        - Estimates damage of every legal move against every opponent.
        - Picks (softmax-sampled, with `temperature`) the highest-damage move.
        - For switch decisions, scores the switch by how much damage the new
          mon could do next turn, scaled by `switch_threshold`.
        - Uses poke-env's calculate_damage, which handles type effectiveness,
          STAB, abilities, items, and stat boosts.

    This player has *no learnable parameters*. It's just a fixed policy.
    The RL agent learning to beat MaxDamage at >50% is a basic sanity check.
    """

    def __init__(
        self,
        battle_format: str = "gen9vgc2023regc",
        switch_threshold: float = 2.0,
        temperature: float = 0.15,
        *args,
        **kwargs,
    ):
        super().__init__(*args, battle_format=battle_format, **kwargs)
        self.switch_threshold = switch_threshold
        self.temperature = temperature

    @staticmethod
    def _estimate_evs_and_nature(
        base_stats: Dict[str, int],
    ) -> Tuple[List[int], str]:
        hp, atk, dfn, spa, spd, spe = (
            base_stats["hp"],
            base_stats["atk"],
            base_stats["def"],
            base_stats["spa"],
            base_stats["spd"],
            base_stats["spe"],
        )

        stat_values = {
            "hp": hp,
            "atk": atk,
            "def": dfn,
            "spa": spa,
            "spd": spd,
            "spe": spe,
        }
        highest = max(stat_values, key=lambda k: stat_values[k])

        evs = [0, 0, 0, 0, 0, 0]

        if highest == "hp":
            evs[0] = 252
            evs[2] = 128
            evs[4] = 128
            nature = "calm"
        elif highest == "atk":
            evs[1] = 252
            if spe >= 90:
                evs[5] = 252
            else:
                evs[0] = 252
            nature = "adamant"
        elif highest == "def":
            evs[2] = 252
            evs[0] = 252
            nature = "bold"
        elif highest == "spa":
            evs[3] = 252
            if spe >= 90:
                evs[5] = 252
            else:
                evs[0] = 252
            nature = "modest"
        elif highest == "spd":
            evs[4] = 252
            evs[0] = 252
            nature = "calm"
        else:
            evs[5] = 252
            if atk >= spa:
                evs[1] = 252
                nature = "adamant"
            else:
                evs[3] = 252
                nature = "modest"

        return evs, nature

    def _estimate_opponent_stats(self, battle) -> None:
        gen_data = GenData.from_gen(battle.gen)

        for mon in battle.opponent_team.values():
            if mon.stats.get("hp") is not None:
                continue

            base_stats = mon.base_stats
            if not base_stats:
                continue

            evs, nature = self._estimate_evs_and_nature(base_stats)
            ivs = [31, 31, 31, 31, 31, 31]
            level = mon.level if mon.level else 50

            try:
                raw_stats = compute_raw_stats(
                    mon.species, evs, ivs, level, nature, gen_data
                )
                mon.stats = {
                    "hp": raw_stats[0],
                    "atk": raw_stats[1],
                    "def": raw_stats[2],
                    "spa": raw_stats[3],
                    "spd": raw_stats[4],
                    "spe": raw_stats[5],
                }
            except Exception:
                continue

    def teampreview(self, battle) -> str:  # type: ignore
        self._estimate_opponent_stats(battle)
        return self._select_teampreview_by_damage(battle)

    def _select_teampreview_by_damage(self, battle) -> str:
        my_team = list(battle.team.values())
        opponent_team = list(battle.opponent_team.values())

        if not my_team or not opponent_team:
            return self.random_teampreview(battle)

        pokemon_total_damages: List[Tuple[Pokemon, float]] = []
        for mon in my_team:
            total_damage = 0.0

            if mon.moves:
                for move in mon.moves.values():
                    if not move or move.base_power == 0:
                        continue

                    for opp_mon in opponent_team:
                        try:
                            damage_range = calculate_damage(
                                mon.identifier(battle.player_role),
                                opp_mon.identifier(battle.opponent_role),
                                move,
                                battle,
                            )
                            if damage_range and damage_range[0] is not None:
                                total_damage += (damage_range[0] + damage_range[1]) / 2.0
                        except Exception:
                            continue

            pokemon_total_damages.append((mon, total_damage))

        pokemon_total_damages.sort(key=lambda x: x[1], reverse=True)

        team_list = list(battle.team.values())
        selected_indices: List[int] = []
        for selected_mon, _ in pokemon_total_damages[:4]:
            for idx, mon in enumerate(team_list):
                if (
                    mon.species == selected_mon.species
                    and (idx + 1) not in selected_indices
                ):
                    selected_indices.append(idx + 1)
                    break

        if len(selected_indices) >= 4:
            return "/team " + "".join(str(i) for i in selected_indices)

        return self.random_teampreview(battle)

    def choose_move(self, battle) -> BattleOrder:  # type: ignore
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        self._estimate_opponent_stats(battle)

        used_switches: Set[str] = set()
        slot_orders: List[BattleOrder] = []

        for slot in range(2):
            active_mon = (
                battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
            )

            if slot < len(battle.force_switch) and battle.force_switch[slot]:
                switches = [
                    s
                    for s in battle.available_switches[slot]
                    if s.species not in used_switches
                ]
                if switches:
                    switch_candidates = [
                        (
                            cast(BattleOrder, self.create_order(s)),
                            self._get_best_move_damage(battle, s)[0],
                        )
                        for s in switches
                    ]
                    chosen_order, _ = self._softmax_sample(
                        switch_candidates, self.temperature
                    )
                    slot_orders.append(chosen_order)
                    chosen_payload = self._get_order_payload(chosen_order)
                    if isinstance(chosen_payload, Pokemon):
                        used_switches.add(chosen_payload.species)
                else:
                    slot_orders.append(DefaultBattleOrder())
                continue

            if any(battle.force_switch):
                slot_orders.append(PassBattleOrder())
                continue

            if active_mon is None:
                slot_orders.append(PassBattleOrder())
                continue

            if slot_is_commanding(battle, slot, battle.last_request):
                slot_orders.append(PassBattleOrder())
                continue

            candidates = self._score_available_actions(battle, slot, used_switches)

            if candidates:
                chosen_order, chosen_score = self._softmax_sample(
                    candidates, self.temperature
                )

                chosen_payload = self._get_order_payload(chosen_order)
                if isinstance(chosen_payload, Pokemon):
                    used_switches.add(chosen_payload.species)

                slot_orders.append(chosen_order)
            else:
                slot_orders.append(DefaultBattleOrder())

        self._apply_mega(slot_orders, battle)

        if len(slot_orders) == 2:
            return DoubleBattleOrder(
                first_order=cast(SingleBattleOrder, slot_orders[0]),
                second_order=cast(SingleBattleOrder, slot_orders[1]),
            )
        elif len(slot_orders) == 1:
            return DoubleBattleOrder(first_order=cast(SingleBattleOrder, slot_orders[0]))

        return self.choose_random_doubles_move(battle)

    @staticmethod
    def _softmax_sample(
        candidates: Sequence[Tuple[BattleOrder, float]], temperature: float
    ) -> Tuple[BattleOrder, float]:
        if not candidates:
            raise ValueError("Cannot sample from empty candidates list")
        if len(candidates) == 1:
            return candidates[0]

        scores = [s for _, s in candidates]

        if temperature <= 0:
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[best_idx]

        max_score = max(scores)
        if max_score > 0:
            normalized = [s / max_score for s in scores]
        else:
            idx = random.randrange(len(candidates))
            return candidates[idx]

        scaled = [s / temperature for s in normalized]
        max_scaled = max(scaled)
        exp_scores = [math.exp(s - max_scaled) for s in scaled]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                return candidates[i]

        return candidates[-1]

    def _score_available_actions(
        self, battle: DoubleBattle, slot: int, used_switches: Set[str]
    ) -> List[Tuple[BattleOrder, float]]:
        if slot_is_commanding(battle, slot, battle.last_request):
            return []

        available_moves = (
            battle.available_moves[slot] if slot < len(battle.available_moves) else []
        )
        active_mon = (
            battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        )
        candidates: List[Tuple[BattleOrder, float]] = []

        request_moves: List[Dict[str, Any]] = []
        if battle.last_request and slot < len(battle.last_request.get("active", [])):
            raw_request_moves = battle.last_request["active"][slot].get("moves", [])
            if isinstance(raw_request_moves, list):
                request_moves = [
                    move for move in raw_request_moves if isinstance(move, dict)
                ]

        request_move_by_id = {
            move["id"]: move
            for move in request_moves
            if isinstance(move.get("id"), str)
            and not move.get("disabled", False)
            and move.get("pp", 1) > 0
        }

        if request_move_by_id:
            available_moves = [
                move for move in available_moves if move.id in request_move_by_id
            ]

        if available_moves and active_mon is not None:
            for move in available_moves:
                request_move = request_move_by_id.get(move.id)
                targets = get_valid_targets(
                    battle,
                    slot,
                    request_move=request_move,
                    move=move,
                    active_mon=active_mon,
                )

                for target in targets:
                    if target < 0:
                        continue

                    target_mon = None
                    if target == 1:
                        opp_active = battle.opponent_active_pokemon
                        target_mon = opp_active[0] if len(opp_active) > 0 else None
                    elif target == 2:
                        opp_active = battle.opponent_active_pokemon
                        target_mon = opp_active[1] if len(opp_active) > 1 else None
                    elif target == 0:
                        for opp in battle.opponent_active_pokemon:
                            if opp is not None:
                                target_mon = opp
                                break

                    if target_mon is None:
                        continue

                    try:
                        player_role = battle.player_role or "p1"
                        opponent_role = battle.opponent_role or "p2"
                        damage_range = calculate_damage(
                            active_mon.identifier(player_role),
                            target_mon.identifier(opponent_role),
                            move,
                            battle,
                        )
                        if damage_range and damage_range[0] is not None:
                            avg_damage = (damage_range[0] + damage_range[1]) / 2.0
                            candidates.append(
                                (
                                    cast(
                                        BattleOrder,
                                        self.create_order(move, move_target=target),
                                    ),
                                    avg_damage,
                                )
                            )
                    except Exception:
                        continue

            if not candidates and available_moves:
                move = available_moves[0]
                request_move = request_move_by_id.get(move.id)
                targets = get_valid_targets(
                    battle,
                    slot,
                    request_move=request_move,
                    move=move,
                    active_mon=active_mon,
                )
                opp_targets = [t for t in targets if t > 0]
                target = (
                    opp_targets[0]
                    if opp_targets
                    else (0 if 0 in targets else targets[0] if targets else 0)
                )
                candidates.append(
                    (cast(BattleOrder, self.create_order(move, move_target=target)), 0.0)
                )

        available_switches = [
            s for s in battle.available_switches[slot] if s.species not in used_switches
        ]
        for switch_mon in available_switches:
            switch_damage = self._get_best_move_damage(battle, switch_mon)[0]
            switch_score = switch_damage / self.switch_threshold
            candidates.append(
                (cast(BattleOrder, self.create_order(switch_mon)), switch_score)
            )

        return candidates

    def _get_best_available_move(
        self, battle: DoubleBattle, slot: int
    ) -> Tuple[Optional[BattleOrder], float]:
        candidates = self._score_available_actions(battle, slot, set())
        if not candidates:
            return (None, 0.0)
        move_candidates = [
            (order, score)
            for order, score in candidates
            if not isinstance(self._get_order_payload(order), Pokemon)
        ]
        if not move_candidates:
            return (None, 0.0)
        best = max(move_candidates, key=lambda x: x[1])
        return best

    def _get_best_move_damage(
        self, battle, attacker: Pokemon
    ) -> Tuple[float, Optional[str], Optional[int]]:
        if not attacker.moves:
            return (0.0, None, None)

        max_damage = 0.0
        best_move_id: Optional[str] = None
        best_target_idx: Optional[int] = None

        targets = [
            (idx, mon)
            for idx, mon in enumerate(battle.opponent_active_pokemon)
            if mon is not None
        ]
        if not targets:
            return (0.0, None, None)

        for move_id, move in attacker.moves.items():
            if not move or move.current_pp == 0:
                continue

            for idx, target in targets:
                try:
                    damage_range = calculate_damage(
                        attacker.identifier(battle.player_role or "p1"),
                        target.identifier(battle.opponent_role or "p2"),
                        move,
                        battle,
                    )
                    if damage_range and damage_range[0] is not None:
                        avg_damage = (damage_range[0] + damage_range[1]) / 2.0
                        if avg_damage >= max_damage:
                            max_damage = avg_damage
                            best_move_id = move_id
                            best_target_idx = idx
                except Exception:
                    continue

        return (max_damage, best_move_id, best_target_idx)

    @staticmethod
    def _get_order_payload(order: BattleOrder):
        return getattr(order, "order", None)

    @staticmethod
    def _apply_mega(slot_orders: List[BattleOrder], battle) -> None:
        """Mega-evolve at most one eligible move slot, in place.

        Mega is strictly beneficial for a max-damage heuristic (the mega forme
        has >= offensive stats), and only one slot may mega per turn, so we mega
        the first slot that (a) can mega-evolve this turn and (b) is making a move
        (not a switch/pass).
        """
        can_mega = getattr(battle, "can_mega_evolve", None) or [False, False]
        for slot, order in enumerate(slot_orders):
            if (
                slot < len(can_mega)
                and can_mega[slot]
                and isinstance(order, SingleBattleOrder)
                and isinstance(MaxDamagePlayer._get_order_payload(order), Move)
            ):
                order.mega = True
                break
