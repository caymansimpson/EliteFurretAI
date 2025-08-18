"""
two_headed_model_deep_dive.py

This script performs a deep dive analysis of a TwoHeadedHybridModel's predictions on battle data.
It loads a model checkpoint and battle files, then iterates through each battle from both
player perspectives, showing the battle state and comparing the model's predictions to actual
player actions.

Usage:
    python two_headed_model_deep_dive.py model_checkpoint.pth battle_files.json
        [--start INDEX]
        [--limit N]
        [--format FORMAT]
        [--feature-set {simple, raw, full}]

    Example: python3 two_headed_model_deep_dive.py model_checkpoint.pth battle_files.json
        --start 0 --limit 10 --format gen9vgc2023regulationc --feature-set full
    Example: python3 src/elitefurretai/scripts/analyze/two_headed_model_deep_dive.py
        data/models/effortless-plasma-11.pth data/battles/supervised_battle_files_w_commander.json
"""

import argparse
import os

import orjson
import torch
from colorama import Fore, Style, init
from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import DoubleBattleOrder

from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.model_utils import MDBO, BattleData, BattleIterator, Embedder
from elitefurretai.utils.battle_order_validator import is_valid_order


# Define the TwoHeadedHybridModel architecture
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.ln = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.LayerNorm(out_features),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.relu(x + residual)  # Add ReLU after addition


class TwoHeadedHybridModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_layers=[1024, 512, 256],
        num_heads=4,
        num_lstm_layers=1,
        num_actions=MDBO.action_space(),
        max_seq_len=17,
        dropout=0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_layers[-1]
        self.num_actions = num_actions

        # Feedforward stack with residual blocks
        layers = []
        prev_size = input_size
        for h in hidden_layers:
            layers.append(ResidualBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.ff_stack = torch.nn.Sequential(*layers)

        # Positional encoding (learned) for the final hidden size
        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        # Bidirectional LSTM
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_proj = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Multihead Self-Attention block
        self.self_attn = torch.nn.MultiheadAttention(
            self.hidden_size, num_heads, batch_first=True
        )

        # Normalize outputs
        self.norm = torch.nn.LayerNorm(self.hidden_size)

        # Output heads
        self.action_head = torch.nn.Linear(self.hidden_size, num_actions)
        self.win_head = torch.nn.Linear(self.hidden_size, 1)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Feedforward stack with residuals
        x = self.ff_stack(x)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = (
            x + self.pos_embedding(positions) * mask.unsqueeze(-1)
            if mask is not None
            else x + self.pos_embedding(positions)
        )

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        # LSTM (packed)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        lstm_out = self.lstm_proj(lstm_out)

        # Multihead Self-Attention
        attn_mask = ~mask.bool()
        attn_out, _ = self.self_attn(
            lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
        )
        out = self.norm(attn_out + lstm_out)

        # *** REMOVE POOLING ***
        # Output heads: now per-step
        action_logits = self.action_head(out)  # (batch, seq_len, num_actions)
        win_logits = self.win_head(out).squeeze(-1)  # (batch, seq_len)

        return action_logits, win_logits

    def predict(self, x, mask=None):
        with torch.no_grad():
            action_logits, win_logits = self.forward(x, mask)
            action_probs = torch.softmax(action_logits, dim=-1)
            win_prob = torch.sigmoid(win_logits)
        return action_probs, win_prob


def format_battle_order(battle_order: MDBO, battle):
    """Format a battle order into a readable string with detailed information."""
    try:
        if battle_order._type == MDBO.TEAMPREVIEW:
            # It's a teampreview order
            team_str = battle_order.message
            # Get the Pokémon species for the chosen positions
            species_info = []
            for char in team_str.replace("/team ", "").strip():
                if char.isdigit():
                    pos = int(char) - 1
                    if pos < len(battle.team) and pos >= 0:
                        species_info.append(list(battle.team.values())[pos].species)

            species_str = ", ".join(species_info)
            return f"{team_str} ({species_str})"
        else:
            # It's a turn or force switch order
            try:
                dbo = battle_order.to_double_battle_order(battle)
                assert isinstance(dbo, DoubleBattleOrder)

                result = []
                # Format each part of the double battle order
                for order in [dbo.first_order, dbo.second_order]:
                    if hasattr(order, "order") and order.order is not None:
                        if isinstance(order.order, Pokemon):
                            # It's a switch
                            text = f"{order.order.species}"
                        elif isinstance(order.order, Move):
                            # It's a move
                            target_text = ""
                            if order.move_target != 0:
                                target_mon = None
                                if order.move_target > 0:
                                    target_idx = order.move_target - 1
                                    if (
                                        target_idx < len(battle.opponent_active_pokemon)
                                        and battle.opponent_active_pokemon[target_idx]
                                        is not None
                                    ):
                                        target_mon = battle.opponent_active_pokemon[
                                            target_idx
                                        ].species
                                else:  # negative target
                                    target_idx = abs(order.move_target) - 1
                                    if (
                                        target_idx < len(battle.active_pokemon)
                                        and battle.active_pokemon[target_idx] is not None
                                    ):
                                        target_mon = battle.active_pokemon[
                                            target_idx
                                        ].species

                                target_text = f", target {order.move_target}"
                                if target_mon:
                                    target_text += f" ({target_mon})"

                            tera_text = ", tera" if order.terastallize else ""
                            text = f"{order.order.id}{target_text}{tera_text}"
                        else:
                            text = str(order)
                    else:
                        text = (
                            "pass"
                            if hasattr(order, "message") and order.message == "pass"
                            else str(order)
                        )
                    result.append(text)

                return f"[{result[0]}], [{result[1]}]"
            except Exception as e:
                return f"{battle_order.message} (Error: {str(e)})"
    except Exception as e:
        return f"{str(battle_order)} (Error: {str(e)})"


def predict_actions(model, state_tensor, battle):
    """Predict actions and their probabilities for a battle state."""
    model.eval()
    with torch.no_grad():
        # Forward pass through the model
        action_logits, win_logits = model(state_tensor.unsqueeze(0))
        win_prob = torch.sigmoid(win_logits).item()

        # Create mask for valid actions based on battle state
        if battle.teampreview:
            # Teampreview mask - all teampreview actions are valid
            action_space = MDBO.teampreview_space()
            mask = torch.ones(action_space, dtype=torch.bool)
            action_logits = action_logits[0, :action_space]
        else:
            # Regular turn - check each action for validity
            action_space = MDBO.action_space()
            mask = torch.zeros(action_space, dtype=torch.bool)
            for i in range(action_space):
                try:
                    action_type = MDBO.TURN
                    if battle.force_switch:
                        action_type = MDBO.FORCE_SWITCH

                    mdbo = MDBO.from_int(i, type=action_type)
                    dbo = mdbo.to_double_battle_order(battle)

                    mask[i] = is_valid_order(dbo, battle)
                except Exception:
                    continue

            action_logits = action_logits[0, :action_space]

        # Apply mask and get probabilities
        masked_logits = action_logits.masked_fill(~mask, float("-inf"))
        action_probs = torch.softmax(masked_logits, dim=0)

        # Get top predictions (up to 10)
        top_k = min(10, int(mask.sum().item()))
        if top_k > 0:
            topk_values, topk_indices = torch.topk(action_probs, top_k)

            predictions = []
            for i, (value, idx) in enumerate(
                zip(topk_values.tolist(), topk_indices.tolist())
            ):
                if battle.teampreview:
                    action = MDBO.from_int(idx, type=MDBO.TEAMPREVIEW)
                else:
                    action_type = MDBO.TURN
                    if battle.force_switch:
                        action_type = MDBO.FORCE_SWITCH
                    action = MDBO.from_int(idx, type=action_type)
                predictions.append((action, value))

            return predictions, win_prob
        else:
            return [], win_prob


def main():
    parser = argparse.ArgumentParser(
        description="Deep dive analysis of TwoHeadedHybridModel predictions."
    )
    parser.add_argument("model_path", type=str, help="Path to model checkpoint.")
    parser.add_argument("battle_filepath", type=str, help="Path to battle data JSON file.")
    parser.add_argument("--start", type=int, default=0, help="Start index for battles.")
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of battles to analyze."
    )
    parser.add_argument("--omniscient", type=bool, default=False, help="Omniscient mode.")
    parser.add_argument(
        "--format", type=str, default="gen9vgc2023regulationc", help="Battle format."
    )
    parser.add_argument(
        "--feature-set", type=str, default="full", help="Embedder feature set (raw, full)."
    )
    args = parser.parse_args()

    # Load model
    print(f"{Fore.CYAN}Loading model from {args.model_path}...{Style.RESET_ALL}")
    embedder = Embedder(
        format=args.format,
        feature_set=args.feature_set.lower(),
        omniscient=args.omniscient,
    )
    model = TwoHeadedHybridModel(embedder.embedding_size)
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))
    model.eval()

    # Load battle data
    print(
        f"{Fore.CYAN}Loading battle data from {args.battle_filepath}...{Style.RESET_ALL}"
    )
    with open(args.battle_filepath, "rb") as f:
        battle_files = orjson.loads(f.read())

    # Limit battles to analyze
    battle_files = battle_files[args.start : args.start + args.limit]
    print(f"{Fore.GREEN}Analyzing {len(battle_files)} battles...{Style.RESET_ALL}\n")

    # Process battles
    for i, battle_file in enumerate(battle_files):
        print(f"\n{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}Battle {i + 1}/{len(battle_files)} - {os.path.basename(battle_file)}{Style.RESET_ALL}"
        )
        print(f"{Fore.YELLOW}{'=' * 80}{Style.RESET_ALL}\n")

        # Load battle data
        try:
            battle_data = BattleData.from_showdown_json(battle_file)
        except Exception as e:
            print(f"{Fore.RED}Error loading battle data: {str(e)}{Style.RESET_ALL}")
            continue

        # Analyze from both perspectives
        for perspective in ["p1", "p2"]:
            print(
                f"\n{Fore.CYAN}{'=' * 40} Perspective: {perspective} {'=' * 40}{Style.RESET_ALL}\n"
            )

            # Initialize battle iterator with this perspective
            iterator = BattleIterator(battle_data, perspective=perspective)

            # Initialize tracking for win probability
            actual_winner = battle_data.winner

            # Step through battle
            turn_count = 0
            while not iterator.battle.finished and iterator.next_input():
                turn_count += 1
                print(
                    f"\n{Fore.BLUE}{'=' * 20} Turn {iterator.battle.turn} {'=' * 20}{Style.RESET_ALL}"
                )

                # Print battle state (using last observation only)
                battle_str = battle_to_str(iterator.battle)
                observations = battle_str.split("\n\n")
                last_observation = observations[-1] if observations else battle_str
                print(f"{Fore.WHITE}{last_observation}{Style.RESET_ALL}")

                # Get model predictions
                try:
                    state_tensor = torch.Tensor(embedder.feature_dict_to_vector(embedder.embed(iterator.battle)))  # type: ignore
                    predictions, win_prob = predict_actions(
                        model, state_tensor, iterator.battle
                    )

                    # Print win probability
                    win_color = Fore.GREEN if win_prob > 0.5 else Fore.RED
                    print(
                        f"\n{win_color}Win probability: {win_prob * 100:.2f}%{Style.RESET_ALL}"
                    )

                    # Print player's actual action
                    try:
                        player_action = iterator.last_order()
                        if player_action:
                            formatted_action = format_battle_order(
                                player_action, iterator.battle
                            )
                            print(
                                f"\n{Fore.GREEN}Player's action: {formatted_action}{Style.RESET_ALL}"
                            )
                        else:
                            print(
                                f"\n{Fore.YELLOW}No player action available for this turn{Style.RESET_ALL}"
                            )
                    except Exception as e:
                        print(
                            f"\n{Fore.RED}Error getting player action: {str(e)}{Style.RESET_ALL}"
                        )

                    # Print model's top predictions
                    print(f"\n{Fore.MAGENTA}Model's top predictions:{Style.RESET_ALL}")
                    if predictions:
                        for j, (pred_action, prob) in enumerate(predictions):
                            color = Fore.YELLOW if j == 0 else Fore.CYAN
                            formatted_pred = format_battle_order(
                                pred_action, iterator.battle
                            )
                            print(
                                f"{color}  {j + 1}. {formatted_pred} - {prob * 100:.2f}%{Style.RESET_ALL}"
                            )
                    else:
                        print(f"{Fore.RED}  No valid predictions{Style.RESET_ALL}")

                except Exception as e:
                    print(f"{Fore.RED}Error during prediction: {str(e)}{Style.RESET_ALL}")

            # Print final results
            print(f"\n{Fore.YELLOW}{'=' * 20} Battle finished {'=' * 20}{Style.RESET_ALL}")

            # Show final win probability
            try:
                final_state = torch.Tensor(embedder.feature_dict_to_vector(embedder.embed(iterator.battle)))  # type: ignore
                _, final_win_prob = predict_actions(model, final_state, iterator.battle)

                is_perspective_winner = actual_winner == perspective
                result_color = Fore.GREEN if is_perspective_winner else Fore.RED
                result_text = "Won" if is_perspective_winner else "Lost"

                print(f"\n{result_color}Actual result: {result_text}{Style.RESET_ALL}")
                print(
                    f"{result_color}Final win probability: {final_win_prob * 100:.2f}%{Style.RESET_ALL}"
                )
            except Exception as e:
                print(
                    f"{Fore.RED}Error calculating final win probability: {str(e)}{Style.RESET_ALL}"
                )


if __name__ == "__main__":
    # Initialize colorama for colored terminal output
    init()

    # Run program
    main()
