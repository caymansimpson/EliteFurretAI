"""Human player that accepts input via CLI for interactive battles."""
from typing import Dict, List, Optional

from poke_env.battle import AbstractBattle, DoubleBattle, Pokemon, Move
from poke_env.player import Player
from poke_env.player.battle_order import (
    BattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    DefaultBattleOrder,
)


class HumanPlayer(Player):
    """
    A player implementation that accepts human input via command-line interface.

    Supports both single and double battles, including team preview, dynamax,
    terastallization, and all standard battle actions.
    """

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Display battle state and prompt user for their action.

        Args:
            battle: The current battle state

        Returns:
            BattleOrder representing the user's chosen action
        """
        print("\n" + "=" * 80)
        print(f"Turn {battle.turn}")
        print("=" * 80)

        self._display_battle_state(battle)

        if battle.teampreview:
            return self._handle_team_preview(battle)

        if isinstance(battle, DoubleBattle):
            return self._handle_doubles_turn(battle)
        else:
            return self._handle_singles_turn(battle)

    def _display_battle_state(self, battle: AbstractBattle) -> None:
        """Display the current state of the battle."""
        if battle.teampreview:
            print("\n=== TEAM PREVIEW ===")
            print("\nYour team:")
            for i, (identifier, pokemon) in enumerate(battle.team.items(), 1):
                print(f"  {i}. {self._format_pokemon(pokemon)}")
            return

        # Display active Pokemon
        print("\n=== ACTIVE POKEMON ===")
        if isinstance(battle, DoubleBattle):
            print("\nYour active Pokemon:")
            for i, pokemon in enumerate(battle.active_pokemon):
                if pokemon:
                    print(f"  Slot {i+1}: {self._format_pokemon(pokemon, detailed=True)}")
                else:
                    print(f"  Slot {i+1}: (fainted)")

            print("\nOpponent's active Pokemon:")
            for i, pokemon in enumerate(battle.opponent_active_pokemon):
                if pokemon:
                    print(f"  Slot {i+1}: {self._format_pokemon(pokemon, detailed=True, opponent=True)}")
                else:
                    print(f"  Slot {i+1}: (fainted)")
        else:
            if battle.active_pokemon:
                print(f"\nYour Pokemon: {self._format_pokemon(battle.active_pokemon, detailed=True)}")
            if battle.opponent_active_pokemon:
                print(f"\nOpponent's Pokemon: {self._format_pokemon(battle.opponent_active_pokemon, detailed=True, opponent=True)}")

        # Display available Pokemon on bench
        available_switches = battle.available_switches
        if available_switches:
            print("\n=== AVAILABLE SWITCHES ===")
            for i, pokemon in enumerate(available_switches, 1):
                print(f"  {i}. {self._format_pokemon(pokemon)}")

    def _format_pokemon(self, pokemon: Pokemon, detailed: bool = False, opponent: bool = False) -> str:
        """Format a Pokemon's information for display."""
        if not pokemon:
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
                boost_str = ", ".join(f"{stat}:{val:+d}" for stat, val in pokemon.boosts.items() if val != 0)
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

    def _handle_team_preview(self, battle: AbstractBattle) -> BattleOrder:
        """Handle team preview selection."""
        team_size = len(battle.team)

        if isinstance(battle, DoubleBattle):
            max_team_size = 4
            print(f"\nSelect {max_team_size} Pokemon for your team (you have {team_size}).")
            print("Enter numbers separated by spaces (e.g., '1 2 3 4'):")
        else:
            print(f"\nOrder your team by entering numbers separated by spaces.")
            print("The first Pokemon will be sent out first.")

        while True:
            try:
                user_input = input("\nYour selection: ").strip()
                if user_input.lower() == 'quit':
                    return ForfeitBattleOrder()

                selections = [int(x) - 1 for x in user_input.split()]

                # Validate selections
                if any(s < 0 or s >= team_size for s in selections):
                    print(f"Error: All numbers must be between 1 and {team_size}")
                    continue

                if len(set(selections)) != len(selections):
                    print("Error: Cannot select the same Pokemon twice")
                    continue

                if isinstance(battle, DoubleBattle):
                    if len(selections) != max_team_size:
                        print(f"Error: Must select exactly {max_team_size} Pokemon")
                        continue
                else:
                    if len(selections) != team_size:
                        print(f"Error: Must order all {team_size} Pokemon")
                        continue

                # Convert to battle order
                team_list = list(battle.team.values())
                ordered_team = [team_list[i] for i in selections]

                return BattleOrder(ordered_team[0], team=ordered_team)

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Please try again.")

    def _handle_singles_turn(self, battle: AbstractBattle) -> BattleOrder:
        """Handle a turn in singles battle."""
        print("\n=== YOUR OPTIONS ===")

        # Display available moves
        available_moves = battle.available_moves
        print("\nMoves:")
        for i, move in enumerate(available_moves, 1):
            print(f"  {i}. {self._format_move(move, battle.active_pokemon)}")

        # Display available switches
        available_switches = battle.available_switches
        if available_switches:
            print("\nSwitches:")
            for i, pokemon in enumerate(available_switches, 1):
                print(f"  s{i}. Switch to {self._format_pokemon(pokemon)}")

        # Special options
        print("\nSpecial:")
        if battle.can_dynamax:
            print("  Add 'd' to dynamax (e.g., '1d' to use move 1 with dynamax)")
        if battle.can_tera:
            print("  Add 't' to terastallize (e.g., '1t' to use move 1 with tera)")
        print("  Type 'quit' to forfeit")

        while True:
            try:
                user_input = input("\nYour action: ").strip().lower()

                if user_input == 'quit':
                    return ForfeitBattleOrder()

                # Parse input
                dynamax = 'd' in user_input
                tera = 't' in user_input
                base_input = user_input.replace('d', '').replace('t', '')

                # Check for switch
                if base_input.startswith('s'):
                    switch_num = int(base_input[1:]) - 1
                    if 0 <= switch_num < len(available_switches):
                        return BattleOrder(available_switches[switch_num])
                    else:
                        print(f"Invalid switch number. Must be between 1 and {len(available_switches)}")
                        continue

                # Otherwise it's a move
                move_num = int(base_input) - 1
                if 0 <= move_num < len(available_moves):
                    move = available_moves[move_num]
                    return BattleOrder(
                        move,
                        dynamax=dynamax and battle.can_dynamax,
                        terastallize=tera and battle.can_tera
                    )
                else:
                    print(f"Invalid move number. Must be between 1 and {len(available_moves)}")
                    continue

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Please try again.")

    def _handle_doubles_turn(self, battle: DoubleBattle) -> DoubleBattleOrder:
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
                print(f"  {i}. {self._format_move(move, battle.active_pokemon[0])}")
        else:
            print("(Fainted - must switch)")

        # Display options for slot 2
        print("\n--- SLOT 2 ---")
        if battle.active_pokemon[1]:
            print("Moves:")
            for i, move in enumerate(available_moves[1], 1):
                print(f"  {i}. {self._format_move(move, battle.active_pokemon[1])}")
        else:
            print("(Fainted - must switch)")

        # Display available switches
        if available_switches:
            print("\n--- AVAILABLE SWITCHES ---")
            for i, pokemon in enumerate(available_switches, 1):
                print(f"  s{i}. {self._format_pokemon(pokemon)}")

        # Display targeting info
        print("\n--- TARGETING ---")
        print("For moves, add target: 1a=Slot1→Opp1, 1b=Slot1→Opp2, 1c=Slot1→Ally")
        print("Example: '1a 2b' = Slot1 attacks opponent's slot1, Slot2 attacks opponent's slot2")

        # Special options
        print("\n--- SPECIAL OPTIONS ---")
        if battle.can_dynamax:
            print("  Add 'd1' or 'd2' to dynamax that slot (e.g., '1a d1 2b')")
        if battle.can_tera:
            print("  Add 't1' or 't2' to terastallize that slot (e.g., '1a t1 2b')")
        print("  Use 's' for switches (e.g., 's1 2a' = switch slot1 to pokemon1, slot2 uses move2)")
        print("  Type 'quit' to forfeit")

        while True:
            try:
                user_input = input("\nYour actions (slot1 slot2): ").strip().lower()

                if user_input == 'quit':
                    return ForfeitBattleOrder()

                # Parse special flags
                dynamax_1 = 'd1' in user_input
                dynamax_2 = 'd2' in user_input
                tera_1 = 't1' in user_input
                tera_2 = 't2' in user_input

                # Remove special flags for parsing
                clean_input = user_input
                for flag in ['d1', 'd2', 't1', 't2']:
                    clean_input = clean_input.replace(flag, '')

                parts = clean_input.split()
                if len(parts) != 2:
                    print("Error: Must provide exactly 2 actions (one for each slot)")
                    continue

                orders = []

                for slot_idx, part in enumerate(parts):
                    # Check if it's a switch
                    if part.startswith('s'):
                        switch_num = int(part[1:]) - 1
                        if 0 <= switch_num < len(available_switches):
                            orders.append(BattleOrder(available_switches[switch_num]))
                        else:
                            print(f"Invalid switch number for slot {slot_idx+1}")
                            orders = None
                            break
                    else:
                        # It's a move - parse move number and target
                        move_num = int(part[0]) - 1
                        target = part[1] if len(part) > 1 else 'a'

                        # Convert target letter to integer
                        target_map = {'a': 1, 'b': 2, 'c': -1}  # -1 for ally
                        if target not in target_map:
                            print(f"Invalid target '{target}'. Use 'a', 'b', or 'c'")
                            orders = None
                            break

                        target_idx = target_map[target]

                        # Validate move number
                        if not battle.active_pokemon[slot_idx]:
                            print(f"Slot {slot_idx+1} Pokemon has fainted, must switch")
                            orders = None
                            break

                        if move_num < 0 or move_num >= len(available_moves[slot_idx]):
                            print(f"Invalid move number for slot {slot_idx+1}")
                            orders = None
                            break

                        move = available_moves[slot_idx][move_num]

                        # Determine dynamax and tera for this slot
                        dynamax = (dynamax_1 if slot_idx == 0 else dynamax_2) and battle.can_dynamax
                        tera = (tera_1 if slot_idx == 0 else tera_2) and battle.can_tera

                        orders.append(BattleOrder(
                            move,
                            move_target=target_idx,
                            dynamax=dynamax,
                            terastallize=tera
                        ))

                if orders is None:
                    continue

                if len(orders) == 2:
                    return DoubleBattleOrder(orders[0], orders[1])

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Please try again.")

    def _format_move(self, move: Move, pokemon: Pokemon) -> str:
        """Format a move for display."""
        move_type = move.type.name if move.type else "???"
        power = move.base_power if move.base_power else "---"
        accuracy = move.accuracy if move.accuracy else "---"
        pp = f"{move.current_pp}/{move.max_pp}" if move.current_pp is not None else "∞"

        return f"{move.id:20s} | Type: {move_type:8s} | Power: {str(power):>3s} | Acc: {str(accuracy):>3s} | PP: {pp}"

    def teampreview(self, battle: AbstractBattle) -> str:
        """
        Handle team preview - delegates to choose_move for consistency.

        This method is called by poke-env during team preview phase.
        """
        order = self.choose_move(battle)
        if isinstance(order, ForfeitBattleOrder):
            return "/forfeit"
        return order.message

    def choose_default_move(self) -> DefaultBattleOrder:
        """Return a default move when no valid moves are available."""
        print("Warning: No valid moves available, using default move")
        return DefaultBattleOrder()
