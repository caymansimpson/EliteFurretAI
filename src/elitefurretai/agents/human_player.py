"""Human player that accepts input via CLI for interactive doubles battles."""

from typing import List, Optional

from poke_env.battle import AbstractBattle, DoubleBattle, Move, Pokemon
from poke_env.player import Player
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    SingleBattleOrder,
)


class HumanPlayer(Player):
    """
    A player implementation that accepts human input via command-line interface.

    Doubles-only: supports team preview, terastallization, switches, and
    standard double-battle actions.
    """

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Display battle state and prompt user for their action.

        Args:
            battle: The current battle state (must be a DoubleBattle)

        Returns:
            BattleOrder representing the user's chosen action
        """
        assert isinstance(battle, DoubleBattle), "HumanPlayer only supports doubles"

        print("\n" + "=" * 80)
        print(f"Turn {battle.turn}")
        print("=" * 80)

        self._display_battle_state(battle)

        if battle.teampreview:
            return self._handle_team_preview(battle)

        return self._handle_doubles_turn(battle)

    def _display_battle_state(self, battle: DoubleBattle) -> None:
        """Display the current state of the battle."""
        if battle.teampreview:
            print("\n=== TEAM PREVIEW ===")
            print("\nYour team:")
            for i, (identifier, pokemon) in enumerate(battle.team.items(), 1):
                print(f"  {i}. {self._format_pokemon(pokemon)}")
            return

        print("\n=== ACTIVE POKEMON ===")
        print("\nYour active Pokemon:")
        for i, pokemon in enumerate(battle.active_pokemon):
            if pokemon:
                print(f"  Slot {i + 1}: {self._format_pokemon(pokemon, detailed=True)}")
            else:
                print(f"  Slot {i + 1}: (fainted)")

        print("\nOpponent's active Pokemon:")
        for i, pokemon in enumerate(battle.opponent_active_pokemon):
            if pokemon:
                print(
                    f"  Slot {i + 1}: {self._format_pokemon(pokemon, detailed=True, opponent=True)}"
                )
            else:
                print(f"  Slot {i + 1}: (fainted)")

    def _format_pokemon(
        self,
        pokemon: Optional[Pokemon],
        detailed: bool = True,
        opponent: bool = False,
    ) -> str:
        """Format a Pokemon's information for display."""
        if pokemon is None:
            return "(None)"

        name = pokemon.species
        if pokemon.item and not opponent:
            name += f" @ {pokemon.item}"

        if detailed:
            hp_percent = int(pokemon.current_hp_fraction * 100)
            hp_bar = self._create_hp_bar(pokemon.current_hp_fraction)
            status = f" [{pokemon.status.name}]" if pokemon.status else ""

            info = f"{name} - {hp_bar} {hp_percent}%{status}"

            # Add active effects
            effects = []
            if pokemon.boosts:
                boost_str = ", ".join(
                    f"{stat}:{val:+d}" for stat, val in pokemon.boosts.items() if val != 0
                )
                if boost_str:
                    effects.append(f"Boosts: {boost_str}")
            if pokemon.effects:
                effect_names = [effect.name for effect in pokemon.effects]
                if effect_names:
                    effects.append(f"Effects: {', '.join(effect_names)}")

            if effects:
                info += "\n      " + " | ".join(effects)

            return info
        else:
            hp_percent = int(pokemon.current_hp_fraction * 100)
            status = f" [{pokemon.status.name}]" if pokemon.status else ""
            return f"{name} - {hp_percent}%{status}"

    def _create_hp_bar(self, hp_fraction: float, width: int = 20) -> str:
        """Create a visual HP bar."""
        filled = int(hp_fraction * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _handle_team_preview(self, battle: DoubleBattle) -> BattleOrder:
        """Handle team preview selection.

        Returns a SingleBattleOrder whose message is the showdown-protocol
        ``/team <indices>`` string, after marking each selected Pokemon's
        ``_selected_in_teampreview`` attribute (required by poke-env).
        """
        team_size = len(battle.team)
        max_team_size = 4

        print(f"\nSelect {max_team_size} Pokemon for your team (you have {team_size}).")
        print("Enter numbers separated by spaces (e.g., '1 2 3 4'):")

        while True:
            try:
                user_input = input("\nYour selection: ").strip()
                if user_input.lower() == "quit":
                    return ForfeitBattleOrder()

                selections = [int(x) - 1 for x in user_input.split()]

                if any(s < 0 or s >= team_size for s in selections):
                    print(f"Error: All numbers must be between 1 and {team_size}")
                    continue

                if len(set(selections)) != len(selections):
                    print("Error: Cannot select the same Pokemon twice")
                    continue

                if len(selections) != max_team_size:
                    print(f"Error: Must select exactly {max_team_size} Pokemon")
                    continue

                team_list = list(battle.team.values())
                for idx in selections:
                    team_list[idx]._selected_in_teampreview = True

                indices = "".join(str(i + 1) for i in selections)
                return SingleBattleOrder(order=f"/team {indices}")

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Please try again.")

    def _handle_doubles_turn(self, battle: DoubleBattle) -> BattleOrder:
        """Handle a turn in doubles battle."""
        print("\n=== YOUR OPTIONS ===")

        # Get available actions for each slot
        available_moves = [battle.available_moves[0], battle.available_moves[1]]
        available_switches = battle.available_switches

        # Display options for slot 1
        print("\n--- SLOT 1 ---")
        if battle.active_pokemon[0]:
            print("Moves:")
            for i, move in enumerate(available_moves[0], 1):
                print(f"  {i}. {self._format_move(move)}")
        else:
            print("(Fainted - must switch)")

        # Display options for slot 2
        print("\n--- SLOT 2 ---")
        if battle.active_pokemon[1]:
            print("Moves:")
            for i, move in enumerate(available_moves[1], 1):
                print(f"  {i}. {self._format_move(move)}")
        else:
            print("(Fainted - must switch)")

        # Display available switches (slot 0's bench is representative for both)
        slot0_switches: List[Pokemon] = list(available_switches[0])
        if slot0_switches:
            print("\n--- AVAILABLE SWITCHES ---")
            for i, pokemon in enumerate(slot0_switches, 1):
                print(f"  s{i}. {self._format_pokemon(pokemon)}")

        # Display targeting info
        print("\n--- TARGETING ---")
        print("For moves, add target: 1a=Slot1→Opp1, 1b=Slot1→Opp2, 1c=Slot1→Ally")
        print(
            "Example: '1a 2b' = Slot1 attacks opponent's slot1, Slot2 attacks opponent's slot2"
        )

        # Special options
        print("\n--- SPECIAL OPTIONS ---")
        if any(battle.can_dynamax):
            print("  Add 'd1' or 'd2' to dynamax that slot (e.g., '1a d1 2b')")
        if any(battle.can_tera):
            print("  Add 't1' or 't2' to terastallize that slot (e.g., '1a t1 2b')")
        print(
            "  Use 's' for switches (e.g., 's1 2a' = switch slot1 to pokemon1, slot2 uses move2)"
        )
        print("  Use 'p' or 'pass' for a slot with no legal action.")
        print("  Type 'quit' to forfeit")

        while True:
            try:
                user_input = input("\nYour actions (slot1 slot2): ").strip().lower()

                if user_input == "quit":
                    return ForfeitBattleOrder()

                # Parse special flags
                dynamax_1 = "d1" in user_input
                dynamax_2 = "d2" in user_input
                tera_1 = "t1" in user_input
                tera_2 = "t2" in user_input

                # Remove special flags for parsing
                clean_input = user_input
                for flag in ["d1", "d2", "t1", "t2"]:
                    clean_input = clean_input.replace(flag, "")

                parts = clean_input.split()
                if len(parts) != 2:
                    print("Error: Must provide exactly 2 actions (one for each slot)")
                    continue

                orders: Optional[List[SingleBattleOrder]] = []

                for slot_idx, part in enumerate(parts):
                    slot_switches = list(available_switches[slot_idx])

                    if part.startswith("s"):
                        switch_num = int(part[1:]) - 1
                        if 0 <= switch_num < len(slot_switches):
                            assert orders is not None
                            orders.append(
                                SingleBattleOrder(order=slot_switches[switch_num])
                            )
                        else:
                            print(f"Invalid switch number for slot {slot_idx + 1}")
                            orders = None
                            break
                    elif part in ("p", "pass"):
                        assert orders is not None
                        orders.append(PassBattleOrder())
                    else:
                        # It's a move - parse move number and target
                        move_num = int(part[0]) - 1
                        target = part[1] if len(part) > 1 else "a"

                        # Convert target letter to integer
                        target_map = {"a": 1, "b": 2, "c": -1}  # -1 for ally
                        if target not in target_map:
                            print(f"Invalid target '{target}'. Use 'a', 'b', or 'c'")
                            orders = None
                            break

                        target_idx = target_map[target]

                        if not battle.active_pokemon[slot_idx]:
                            print(f"Slot {slot_idx + 1} Pokemon has fainted, must switch")
                            orders = None
                            break

                        if move_num < 0 or move_num >= len(available_moves[slot_idx]):
                            print(f"Invalid move number for slot {slot_idx + 1}")
                            orders = None
                            break

                        move = available_moves[slot_idx][move_num]

                        dynamax = (
                            dynamax_1 if slot_idx == 0 else dynamax_2
                        ) and battle.can_dynamax[slot_idx]
                        tera = (tera_1 if slot_idx == 0 else tera_2) and battle.can_tera[
                            slot_idx
                        ]

                        assert orders is not None
                        orders.append(
                            SingleBattleOrder(
                                order=move,
                                move_target=target_idx,
                                dynamax=dynamax,
                                terastallize=tera,
                            )
                        )

                if orders is None:
                    continue

                if len(orders) == 2:
                    return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Please try again.")

    def _format_move(self, move: Move) -> str:
        """Format a move for display."""
        move_type = move.type.name if move.type else "???"
        power = move.base_power if move.base_power else "---"
        accuracy = move.accuracy if move.accuracy else "---"
        pp = f"{move.current_pp}/{move.max_pp}" if move.current_pp is not None else "∞"

        return f"{move.id:20s} | Type: {move_type:8s} | Power: {str(power):>3s} | Acc: {str(accuracy):>3s} | PP: {pp}"

    def teampreview(self, battle: AbstractBattle) -> str:
        """Handle team preview by delegating to choose_move for consistency."""
        order = self.choose_move(battle)
        return order.message

    @staticmethod
    def choose_default_move() -> DefaultBattleOrder:
        """Return a default move when no valid moves are available."""
        print("Warning: No valid moves available, using default move")
        return DefaultBattleOrder()
