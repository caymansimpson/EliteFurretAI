#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay battles from Showdown JSON logs and analyze model predictions.

This script loads a battle from raw Showdown JSON, replays it step-by-step,
and shows model predictions vs. actual actions taken at each decision point.

Usage:
    python behavior_clone_replay.py \\
        --battle-file data/battles/gen9vgc2023regc_raw/2023-01-gen9vgc2023regc-1500.json \\
        --teampreview-model-path data/models/teampreview_model.pt \\
        --action-model-path data/models/action_model.pt \\
        --win-model-path data/models/win_model.pt \\
        --perspective p1 \\
        --verbose

Features:
    - Uses BCPlayer with three separate models (teampreview, action, win)
    - Top-5 action predictions with probabilities
    - Win prediction at each turn
    - Highlights when actual action is outside top-5
    - Shows battle state at each decision point
    - Supports both teampreview and turn actions
    - Configs are embedded in .pt model files
"""

import argparse
import json
from typing import Dict, Union

import torch
from poke_env.battle import DoubleBattle, Move, Pokemon
from poke_env.player import BattleOrder, DoubleBattleOrder

from elitefurretai.etl.battle_data import BattleData
from elitefurretai.etl.battle_iterator import BattleIterator
from elitefurretai.etl.encoder import MDBO
from elitefurretai.supervised.behavior_clone_player import BCPlayer


def format_action_human_readable(
    action: Union[str, BattleOrder, MDBO], battle: DoubleBattle
) -> str:
    """Format action with human-readable move names, Pokemon names, and targets."""

    # Handle string messages (teampreview or battle order messages)
    if isinstance(action, str):
        # Teampreview format: "/team 1234"
        if action.startswith("/team "):
            team_order = action.replace("/team ", "")
            # Get Pokemon names from teampreview team
            pokemon_names = []
            for pos_char in team_order:
                try:
                    pos = int(pos_char) - 1  # Convert to 0-indexed
                    if 0 <= pos < len(battle.teampreview_team):
                        pokemon_names.append(battle.teampreview_team[pos].species)
                    else:
                        pokemon_names.append(f"?{pos_char}")
                except (ValueError, IndexError):
                    pokemon_names.append(f"?{pos_char}")
            return f"Team: {' / '.join(pokemon_names)}"
        # Battle order message format: "/choose move1, move2" or "/choose pass, switch X"
        else:
            # Just return the message as-is for now - it's already somewhat readable
            return action

    # Handle MDBO - convert to DoubleBattleOrder first
    if isinstance(action, MDBO):
        if battle.teampreview:
            team_order = action.message.replace("/team ", "")
            # Get Pokemon names from teampreview team
            pokemon_names = []
            for pos_char in team_order:
                try:
                    pos = int(pos_char) - 1  # Convert to 0-indexed
                    if 0 <= pos < len(battle.teampreview_team):
                        pokemon_names.append(battle.teampreview_team[pos].species)
                    else:
                        pokemon_names.append(f"?{pos_char}")
                except (ValueError, IndexError):
                    pokemon_names.append(f"?{pos_char}")
            return f"Team: {' / '.join(pokemon_names)}"
        try:
            action = action.to_double_battle_order(battle)
        except Exception as e:
            return f"[Error converting MDBO: {e}]"

    # Handle DoubleBattleOrder
    if isinstance(action, DoubleBattleOrder):
        parts = []

        for i, order in enumerate([action.first_order, action.second_order]):
            if order is None:
                parts.append("Pass")
                continue

            # Get the Pokemon making the action (from active_pokemon list)
            active_mon = (
                battle.active_pokemon[i] if i < len(battle.active_pokemon) else None
            )
            mon_name = active_mon.species if active_mon else f"Mon{i + 1}"

            # Handle switch
            if isinstance(order.order, Pokemon):
                parts.append(f"{mon_name} → Switch to {order.order.species}")

            # Handle move
            elif isinstance(order.order, Move):
                move_str = f"{mon_name}: {order.order.id.title()}"

                # Add target information
                target_str = ""
                if order.move_target == DoubleBattle.OPPONENT_1_POSITION:
                    opp_mon = battle.opponent_active_pokemon[0]
                    target_str = f" → Opp {opp_mon.species if opp_mon else 'Mon1'}"
                elif order.move_target == DoubleBattle.OPPONENT_2_POSITION:
                    opp_mon = battle.opponent_active_pokemon[1]
                    target_str = f" → Opp {opp_mon.species if opp_mon else 'Mon2'}"
                elif order.move_target == DoubleBattle.POKEMON_1_POSITION:
                    ally_mon = battle.active_pokemon[0]
                    target_str = f" → {ally_mon.species if ally_mon else 'Mon1'}"
                elif order.move_target == DoubleBattle.POKEMON_2_POSITION:
                    ally_mon = battle.active_pokemon[1]
                    target_str = f" → {ally_mon.species if ally_mon else 'Mon2'}"
                elif order.move_target == -1:
                    target_str = " (Self)"
                elif order.move_target == -2:
                    target_str = " (Ally)"

                move_str += target_str

                # Add battle mechanics
                if order.terastallize:
                    move_str += " [TERA]"
                if order.dynamax:
                    move_str += " [DMAX]"
                if order.mega:
                    move_str += " [MEGA]"
                if order.z_move:
                    move_str += " [Z-MOVE]"

                parts.append(move_str)
            else:
                parts.append(f"{mon_name}: {order.order}")

        return " | ".join(parts)

    # Fallback
    return str(action)


def format_action(action, battle: DoubleBattle) -> str:
    """Format action as human-readable string."""
    if isinstance(action, str):
        return f"Team: {action}"
    elif isinstance(action, MDBO):
        if battle.teampreview:
            return f"Team: {action.message}"
        else:
            try:
                dbo = action.to_double_battle_order(battle)
                return str(dbo)
            except Exception as e:
                return f"MDBO({action.to_int()}): {e}"
    else:
        return str(action)


def format_current_battle_state(battle: DoubleBattle) -> str:
    """Format current battle state without history."""
    message = ""

    # For teampreview, just show the team rosters
    if battle.teampreview:
        message += (
            f"  My Team: [{', '.join(mon.species for mon in battle.teampreview_team)}]"
        )
        message += f"\n  Opp Team: [{', '.join(mon.species for mon in battle.teampreview_opponent_team)}]"
        return message

    # Active Pokemon
    active = [mon for mon in battle.active_pokemon if mon is not None]
    opp_active = [mon for mon in battle.opponent_active_pokemon if mon is not None]

    message += f"  My Active:  [{', '.join(f'{mon.species} ({mon.current_hp}/{mon.max_hp} HP)' for mon in active)}]"
    message += f"\n  Opp Active: [{', '.join(f'{mon.species} ({mon.current_hp}/{mon.max_hp} HP)' for mon in opp_active)}]"

    # Battle Conditions
    if len(battle.weather) > 0:
        message += f"\n  Weather: [{', '.join(w.name for w in battle.weather)}]"
    if len(battle.fields) > 0:
        message += f"\n  Fields: [{', '.join(f.name for f in battle.fields)}]"
    if len(battle.side_conditions) > 0:
        message += f"\n  My Side Conditions: [{', '.join(sc.name for sc in battle.side_conditions)}]"
    if len(battle.opponent_side_conditions) > 0:
        message += f"\n  Opp Side Conditions: [{', '.join(sc.name for sc in battle.opponent_side_conditions)}]"

    # Team status with boosts and effects
    message += "\n  My Team:"
    for ident, mon in battle.team.items():
        status_str = f"{mon.status.name}" if mon.status else "OK"
        message += f"\n    {mon.species}: {mon.current_hp}/{mon.max_hp} HP, {status_str}"
        if mon.item:
            message += f", Item: {mon.item}"
        # Add boosts
        active_boosts = {k: v for k, v in mon.boosts.items() if v != 0}
        if active_boosts:
            message += f", Boosts: {active_boosts}"
        # Add effects
        if mon.effects:
            message += f", Effects: [{', '.join(e.name for e in mon.effects)}]"

    message += "\n  Opp Team:"
    for ident, mon in battle.opponent_team.items():
        status_str = f"{mon.status.name}" if mon.status else "OK"
        message += f"\n    {mon.species}: {mon.current_hp}/{mon.max_hp} HP, {status_str}"
        if mon.item:
            message += f", Item: {mon.item}"
        # Add boosts
        active_boosts = {k: v for k, v in mon.boosts.items() if v != 0}
        if active_boosts:
            message += f", Boosts: {active_boosts}"
        # Add effects
        if mon.effects:
            message += f", Effects: [{', '.join(e.name for e in mon.effects)}]"

    return message


def replace_nicknames_in_log(log_line: str, battle: DoubleBattle) -> str:
    """Replace Pokemon nicknames with species names in battle log lines."""
    # Pattern: "p1a: Name" or "p2b: Name" etc.
    import re

    def replace_identifier(match):
        identifier = match.group(0)  # e.g., "p1a: Incineroar"
        try:
            # Try to get the Pokemon from battle
            pokemon = battle.get_pokemon(identifier)
            if pokemon:
                # Replace with species name
                prefix = identifier.split(":")[0]  # "p1a" or "p2b"
                return f"{prefix}: {pokemon.species}"
        except (ValueError, KeyError, AttributeError):
            # If Pokemon not found or error, return original
            pass
        return identifier

    # Match pattern like "p1a: Name" or "p2b: Name"
    pattern = r"p[12][ab]:\s*[^|,\]]*"
    result = re.sub(pattern, replace_identifier, log_line)
    return result


def analyze_battle(
    battle_file: str,
    player: BCPlayer,
    perspective: str = "p1",
    verbose: bool = False,
):
    """Replay battle and analyze model predictions at each step."""
    # Load battle data
    with open(battle_file, "r") as f:
        battle_json = json.load(f)

    battle_data = BattleData.from_showdown_json(battle_json)

    print(f"\n{'=' * 80}")
    print(f"Battle: {battle_data.battle_tag}")
    print(f"Format: {battle_data.format}")
    print(
        f"Winner: {battle_data.winner} ({'p1' if battle_data.winner == battle_data.p1 else 'p2'})"
    )
    print(f"Perspective: {perspective}")
    print(f"{'=' * 80}\n")

    # Create iterator
    iterator = BattleIterator(battle_data, perspective=perspective, omniscient=False)

    good_predictions = 0
    bad_predictions = 0
    last_index = 0  # Track where we were in the logs

    try:
        while iterator.next_input():
            # Type assert for proper typing
            battle = iterator.battle
            assert isinstance(battle, DoubleBattle)

            # Print battle logs that happened since last iteration
            current_index = iterator._index
            if current_index > last_index:
                print("\n[Continuing Battle Logs to next Decision...]")
                for log_line in battle_data.logs[last_index:current_index]:
                    # Replace nicknames with species names
                    formatted_log = replace_nicknames_in_log(log_line, battle)
                    print(f"  {formatted_log}")
            last_index = current_index

            # Embed state and add to trajectory
            state_vec = player.embed_battle_state(battle)
            if battle.battle_tag not in player._trajectories:
                player._trajectories[battle.battle_tag] = []
            player._trajectories[battle.battle_tag].append(state_vec)

            # Get ground truth action
            actual_action = iterator.last_order()

            # Get model predictions using BCPlayer, passing the correct action type from iterator
            traj = torch.Tensor(player._trajectories[battle.battle_tag]).unsqueeze(0)
            action_probs: Dict[BattleOrder, float] = player.predict(
                traj, battle, action_type=iterator.last_input_type
            )  # type: ignore

            # Get win prediction from win model
            traj_win = traj[:, -player.win_model.max_seq_len :, :]
            player.win_model.eval()
            with torch.no_grad():
                _, _, win_logits = player.win_model(traj_win)
                win_pred = float(win_logits[0, -1].item())

            # Sort by probability
            sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)

            # Find actual action rank
            actual_rank = None
            # Convert actual_action (MDBO) to comparable format
            actual_msg = None
            if battle.teampreview:
                actual_key = actual_action.message
            else:
                try:
                    actual_order = actual_action.to_double_battle_order(battle)
                    # Try using the object first
                    actual_key = actual_order  # type: ignore
                    actual_msg = actual_order.message  # Fallback to message
                except Exception:
                    actual_key = None

            for rank, (action, prob) in enumerate(sorted_actions, 1):
                # Compare actions
                if battle.teampreview:
                    # For teampreview, both should be strings (messages)
                    if action == actual_key:
                        actual_rank = rank
                        break
                else:
                    # For turn actions, try object comparison then message comparison
                    if action == actual_key or (
                        actual_msg and isinstance(action, str) and action == actual_msg
                    ):
                        actual_rank = rank
                        break
                    # Also handle case where action is DoubleBattleOrder and we need to compare messages
                    if (
                        hasattr(action, "message")
                        and actual_msg
                        and action.message == actual_msg
                    ):
                        actual_rank = rank
                        break

            is_good_pred = actual_rank is not None and actual_rank <= 5
            if is_good_pred:
                good_predictions += 1
            else:
                bad_predictions += 1

            # Display turn header
            if battle.teampreview:
                turn_type = "TEAMPREVIEW"
            else:
                # Check if this is a force switch (requesting input within the same turn)
                is_force_switch = any(battle.force_switch)
                turn_type = f"TURN {battle.turn}" + (
                    " [FORCE SWITCH]" if is_force_switch else ""
                )

            pred_quality = "✓ GOOD" if is_good_pred else "✗ BAD"

            print(f"\n{'─' * 80}")
            print(f"{turn_type} | Win Prediction: {win_pred:+.3f} | {pred_quality}")
            print(f"{'─' * 80}")

            print("\n[Battle State]")
            print(format_current_battle_state(battle))
            print()

            # Show all predictions sorted by probability
            print(f"[All Predictions - {len(sorted_actions)} valid actions]")
            for rank, (action, prob) in enumerate(sorted_actions, 1):
                action_str = format_action_human_readable(action, battle)
                # Check if this is the actual action
                is_actual = False
                if isinstance(action, str) and isinstance(actual_action, MDBO):
                    is_actual = action == actual_action.message
                else:
                    is_actual = action == actual_action
                marker = "★" if is_actual else " "
                if rank <= 5:
                    print(f"  {rank:3d}. {marker} {prob:6.2%} | {action_str}")

            print(
                f"\n  Actual action: {format_action_human_readable(actual_action, battle)}  (Rank #{actual_rank})"
            )

    except StopIteration:
        pass

    # Summary
    print(f"\n{'=' * 80}")
    print("Battle Complete")
    print(f"{'=' * 80}")
    print(
        f"Good Predictions (actual in top-5): {good_predictions}/{good_predictions + bad_predictions} ({100 * good_predictions / (good_predictions + bad_predictions):.1f}%)"
    )
    print(
        f"Bad Predictions (actual not in top-5): {bad_predictions}/{good_predictions + bad_predictions} ({100 * bad_predictions / (good_predictions + bad_predictions):.1f}%)"
    )
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay battles and analyze model predictions using BCPlayer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--battle-file",
        type=str,
        required=True,
        help="Path to raw Showdown JSON battle file",
    )

    parser.add_argument(
        "--unified-model-path",
        type=str,
        help="Path to unified model checkpoint (.pt file with all three heads)",
    )

    parser.add_argument(
        "--teampreview-model-path",
        type=str,
        help="Path to teampreview model checkpoint (.pt file with embedded config)",
    )

    parser.add_argument(
        "--action-model-path",
        type=str,
        help="Path to action model checkpoint (.pt file with embedded config)",
    )

    parser.add_argument(
        "--win-model-path",
        type=str,
        help="Path to win model checkpoint (.pt file with embedded config)",
    )

    parser.add_argument(
        "--perspective",
        type=str,
        choices=["p1", "p2"],
        default="p1",
        help="Which player's perspective to analyze",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Show full battle state at each turn"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on",
    )

    parser.add_argument(
        "--battle-format",
        type=str,
        default="gen9vgc2023regc",
        help="Battle format (default: gen9vgc2023regc)",
    )

    args = parser.parse_args()

    # Validate model path arguments
    has_unified = args.unified_model_path is not None
    has_separate = all(
        [
            args.teampreview_model_path is not None,
            args.action_model_path is not None,
            args.win_model_path is not None,
        ]
    )
    has_any_separate = any(
        [
            args.teampreview_model_path is not None,
            args.action_model_path is not None,
            args.win_model_path is not None,
        ]
    )

    if has_unified and has_any_separate:
        parser.error("Cannot specify both --unified-model-path and individual model paths")
    if not has_unified and not has_separate:
        parser.error(
            "Must specify either --unified-model-path or all three individual model paths (--teampreview-model-path, --action-model-path, --win-model-path)"
        )

    # Create BCPlayer with unified or separate models
    print("Loading models...")
    if has_unified:
        player = BCPlayer(
            unified_model_filepath=args.unified_model_path,
            battle_format=args.battle_format,
            device=args.device,
            probabilistic=False,  # Use greedy selection for analysis
            verbose=True,  # Show initialization progress
        )
    else:
        player = BCPlayer(
            teampreview_model_filepath=args.teampreview_model_path,
            action_model_filepath=args.action_model_path,
            win_model_filepath=args.win_model_path,
            battle_format=args.battle_format,
            device=args.device,
            probabilistic=False,  # Use greedy selection for analysis
            verbose=True,  # Show initialization progress
        )

    # Analyze battle
    analyze_battle(
        args.battle_file,
        player,
        perspective=args.perspective,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
