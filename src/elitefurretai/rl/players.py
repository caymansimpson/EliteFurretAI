import logging
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple, cast

import torch
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.calc import calculate_damage
from poke_env.data import GenData
from poke_env.player import BattleOrder, DoubleBattleOrder, Player
from poke_env.player.battle_order import (
    DefaultBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)
from poke_env.stats import compute_raw_stats

from elitefurretai.rl.multiprocess_actor import (
    BatchInferencePlayer,
    cleanup_worker_executors,
)
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel

logger = logging.getLogger("MaxDamagePlayer")


class RNaDAgent(torch.nn.Module):
    """
    RL Agent wrapper around FlexibleThreeHeadedModel.
    Handles hidden states and value function transformation.
    """

    def __init__(self, model: FlexibleThreeHeadedModel):
        super().__init__()
        self.model = model

    def get_initial_state(self, batch_size, device):
        num_directions = 2
        h = torch.zeros(
            self.model.lstm.num_layers * num_directions,
            batch_size,
            self.model.lstm_hidden_size,
            device=device,
        )
        c = torch.zeros(
            self.model.lstm.num_layers * num_directions,
            batch_size,
            self.model.lstm_hidden_size,
            device=device,
        )
        return (h, c)

    def forward(self, x, hidden_state=None, mask=None):
        turn_logits, tp_logits, value, next_hidden = self.model.forward_with_hidden(
            x, hidden_state, mask
        )
        return turn_logits, tp_logits, value, next_hidden


class MaxDamagePlayer(Player):
    def __init__(
        self,
        battle_format: str = "gen9vgc2023regc",
        switch_threshold: float = 1.5,
        temperature: float = 0.5,
        debug: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, battle_format=battle_format, **kwargs)
        self.switch_threshold = switch_threshold
        self.temperature = temperature
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("[MDP] %(message)s"))
                logger.addHandler(handler)

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
                if mon.species == selected_mon.species and (idx + 1) not in selected_indices:
                    selected_indices.append(idx + 1)
                    break

        if len(selected_indices) >= 4:
            return "/team " + "".join(str(i) for i in selected_indices)

        return self.random_teampreview(battle)

    def choose_move(self, battle) -> BattleOrder:  # type: ignore
        if not isinstance(battle, DoubleBattle):
            return self.choose_random_move(battle)

        self._estimate_opponent_stats(battle)

        if self.debug:
            active = [m.species if m else "None" for m in battle.active_pokemon]
            opp_active = [m.species if m else "None" for m in battle.opponent_active_pokemon]
            logger.debug(f"Turn {battle.turn}: Active={active} vs Opp={opp_active}")
            for m in battle.opponent_active_pokemon:
                if m:
                    logger.debug(f"  Opp {m.species} stats={m.stats}")

        used_switches: Set[str] = set()
        slot_orders: List[BattleOrder] = []

        for slot in range(2):
            active_mon = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None

            if slot < len(battle.force_switch) and battle.force_switch[slot]:
                switches = [
                    s for s in battle.available_switches[slot]
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
                    chosen_order, _ = self._softmax_sample(switch_candidates, self.temperature)
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

            candidates = self._score_available_actions(battle, slot, used_switches)

            if candidates:
                chosen_order, chosen_score = self._softmax_sample(candidates, self.temperature)

                chosen_payload = self._get_order_payload(chosen_order)
                if isinstance(chosen_payload, Pokemon):
                    used_switches.add(chosen_payload.species)

                slot_orders.append(chosen_order)
                if self.debug:
                    if isinstance(chosen_payload, Pokemon):
                        order_desc = f"SWITCH to {chosen_payload.species}"
                    elif chosen_payload is not None and hasattr(chosen_payload, "id"):
                        order_desc = (
                            f"{chosen_payload.id} "
                            f"target={getattr(chosen_order, 'move_target', None)}"
                        )
                    else:
                        order_desc = str(chosen_order)
                    logger.debug(
                        f"  Slot {slot} ({active_mon.species}): {order_desc}"
                        f" score={chosen_score:.0f} (from {len(candidates)} candidates, temp={self.temperature})"
                    )
            else:
                slot_orders.append(DefaultBattleOrder())
                if self.debug:
                    logger.debug(f"  Slot {slot} ({active_mon.species if active_mon else 'None'}): DEFAULT (no valid move or switch)")

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
        available_moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
        active_mon = battle.active_pokemon[slot] if slot < len(battle.active_pokemon) else None
        candidates: List[Tuple[BattleOrder, float]] = []

        if available_moves and active_mon is not None:
            for move in available_moves:
                targets = battle.get_possible_showdown_targets(move, active_mon)

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
                            if self.debug:
                                logger.debug(
                                    f"    {active_mon.species} {move.id} -> {target_mon.species}"
                                    f" (target={target}): {damage_range[0]}-{damage_range[1]}"
                                    f" (avg={avg_damage:.0f})"
                                )
                            candidates.append(
                                (
                                    cast(BattleOrder, self.create_order(move, move_target=target)),
                                    avg_damage,
                                )
                            )
                    except Exception as e:
                        if self.debug:
                            logger.debug(
                                f"    {active_mon.species} {move.id} -> {target_mon.species}"
                                f" (target={target}): EXCEPTION: {e}"
                            )
                        continue

            if not candidates and available_moves:
                move = available_moves[0]
                targets = battle.get_possible_showdown_targets(move, active_mon)
                opp_targets = [t for t in targets if t > 0]
                target = opp_targets[0] if opp_targets else (0 if 0 in targets else targets[0] if targets else 0)
                candidates.append(
                    (cast(BattleOrder, self.create_order(move, move_target=target)), 0.0)
                )
                if self.debug:
                    logger.debug(
                        f"    {active_mon.species}: FALLBACK {move.id} target={target}"
                        f" (all_targets={targets})"
                    )

        available_switches = [
            s for s in battle.available_switches[slot]
            if s.species not in used_switches
        ]
        for switch_mon in available_switches:
            switch_damage = self._get_best_move_damage(battle, switch_mon)[0]
            switch_score = switch_damage / self.switch_threshold
            candidates.append((cast(BattleOrder, self.create_order(switch_mon)), switch_score))

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


__all__ = [
    "RNaDAgent",
    "MaxDamagePlayer",
    "BatchInferencePlayer",
    "cleanup_worker_executors",
]
