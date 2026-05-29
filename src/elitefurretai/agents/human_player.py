"""Human player that accepts input via CLI for interactive doubles battles."""

from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)

from elitefurretai.agents._human_action_parser import (
    ActionParseError,
    parse_action,
)
from elitefurretai.engine.battle_renderer import (
    format_action_reference,
    format_battle_state,
    format_pokemon_line,
    format_teampreview,
)


class HumanPlayer(Player):
    """A CLI player for interactive doubles battles.

    Reads action lines from stdin per the grammar documented in
    [_human_action_parser.py](_human_action_parser.py). State rendering
    lives in [engine.battle_renderer](../engine/battle_renderer.py); this
    class only handles I/O, prompting, and orchestration.
    """

    _action_reference_shown: bool = False

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle), "HumanPlayer only supports doubles"

        if not self._action_reference_shown:
            print(format_action_reference())
            self._action_reference_shown = True

        if battle.teampreview:
            return self._handle_team_preview(battle)

        print(format_battle_state(battle))

        if any(battle.force_switch):
            return self._handle_force_switch(battle)

        return self._handle_action_turn(battle)

    def _handle_team_preview(self, battle: DoubleBattle) -> BattleOrder:
        """Prompt for the four-mon team order, return ``/team <digits>``."""
        print(format_teampreview(battle))

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

    def _handle_action_turn(self, battle: DoubleBattle) -> BattleOrder:
        """Prompt for a normal-turn action and parse it."""
        while True:
            user_input = input("\nYour actions: ").strip()
            try:
                return parse_action(user_input, battle)
            except ActionParseError as e:
                print(f"Invalid action: {e}")

    def _handle_force_switch(self, battle: DoubleBattle) -> BattleOrder:
        """Prompt for force-switch input, parse it with the force_switch flag set."""
        print("\n--- FORCE SWITCH REQUIRED ---")
        forced = list(battle.force_switch)
        if sum(forced) == 1:
            forced_idx = 0 if forced[0] else 1
            fainted = battle.active_pokemon[forced_idx]
            species = fainted.species if fainted is not None else "(unknown)"
            print(
                f"Your slot {forced_idx + 1} must switch ({species}). "
                f"Slot {2 - forced_idx} takes no action this prompt."
            )
        else:
            print("Both of your slots must switch.")

        print("\nBench:")
        for mon in battle.available_switches[0]:
            print(f"  {format_pokemon_line(mon, is_opponent=False)}")

        while True:
            prompt = (
                "\nYour action (just the switch): "
                if sum(forced) == 1
                else "\nYour actions (slot1, slot2): "
            )
            user_input = input(prompt).strip()
            try:
                return parse_action(user_input, battle, force_switch=forced)
            except ActionParseError as e:
                print(f"Invalid action: {e}")

    def teampreview(self, battle: AbstractBattle) -> str:
        """poke-env's teampreview hook — delegate to choose_move for consistency."""
        order = self.choose_move(battle)
        return order.message

    @staticmethod
    def choose_default_move() -> DefaultBattleOrder:
        print("Warning: No valid moves available, using default move")
        return DefaultBattleOrder()


__all__ = ["HumanPlayer"]
