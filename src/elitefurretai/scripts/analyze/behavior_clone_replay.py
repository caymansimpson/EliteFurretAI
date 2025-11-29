#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay battles from Showdown JSON logs and analyze model predictions.

This script loads a battle from raw Showdown JSON, replays it step-by-step,
and shows model predictions vs. actual actions taken at each decision point.

Usage:
    python behavior_clone_replay.py \\
        --battle-file data/battles/gen9vgc2023regulationc_raw/2023-01-gen9vgc2023regulationc-1500.json \\
        --model-path data/models/three_headed_model.pt \\
        --config-path data/models/three_headed_config.json \\
        --perspective p1 \\
        --verbose

Features:
    - Top-5 action predictions with probabilities
    - Win prediction at each turn
    - Highlights when actual action is outside top-5
    - Shows battle state at each decision point
    - Supports both teampreview and turn actions
"""

import argparse
import json
from typing import Dict, List, Tuple, Optional

import torch
from poke_env.battle import DoubleBattle, AbstractBattle

from elitefurretai.model_utils.battle_data import BattleData
from elitefurretai.model_utils.battle_iterator import BattleIterator
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.encoder import MDBO
from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.utils.battle_order_validator import is_valid_order


# Copy FlexibleThreeHeadedModel architecture from training script
class ResidualBlock(torch.nn.Module):
    """Residual block without second ReLU to allow negative values."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.ln = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        torch.nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.linear.bias, 0)

        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            shortcut_linear = torch.nn.Linear(in_features, out_features)
            torch.nn.init.kaiming_normal_(shortcut_linear.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(shortcut_linear.bias, 0)
            self.shortcut = torch.nn.Sequential(
                shortcut_linear,
                torch.nn.LayerNorm(out_features),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x + residual


class GroupedFeatureEncoder(torch.nn.Module):
    """Encodes features by semantic groups with cross-attention for Pokemon."""
    def __init__(self, group_sizes, hidden_dim=128, aggregated_dim=1024, dropout=0.1, pokemon_attention_heads=2):
        super().__init__()
        self.group_sizes = group_sizes
        self.hidden_dim = hidden_dim

        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(size, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            for size in group_sizes
        ])

        for encoder in self.encoders:
            linear_layer = encoder[0]  # type: ignore
            torch.nn.init.kaiming_normal_(linear_layer.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(linear_layer.bias, 0)

        self.pokemon_cross_attn = torch.nn.MultiheadAttention(
            hidden_dim, num_heads=pokemon_attention_heads, batch_first=True, dropout=dropout
        )
        self.pokemon_norm = torch.nn.LayerNorm(hidden_dim)

        self.aggregator = torch.nn.Linear(hidden_dim * len(group_sizes), aggregated_dim)
        torch.nn.init.xavier_normal_(self.aggregator.weight)
        torch.nn.init.constant_(self.aggregator.bias, 0)

    def forward(self, x):
        batch, seq, _ = x.shape

        group_features = []
        start_idx = 0
        for encoder, size in zip(self.encoders, self.group_sizes):
            group = x[:, :, start_idx:start_idx + size]
            group_features.append(encoder(group))
            start_idx += size

        player_pokemon = torch.stack(group_features[:6], dim=2)
        player_pokemon_flat = player_pokemon.reshape(batch * seq, 6, -1)

        attn_out, _ = self.pokemon_cross_attn(
            player_pokemon_flat,
            player_pokemon_flat,
            player_pokemon_flat
        )

        attn_out = attn_out.reshape(batch, seq, 6, -1)

        for i in range(6):
            group_features[i] = self.pokemon_norm(group_features[i] + attn_out[:, :, i, :])

        concatenated = torch.cat(group_features, dim=-1)
        return self.aggregator(concatenated)


class FlexibleThreeHeadedModel(torch.nn.Module):
    """Three-headed model with teampreview, turn action, and win prediction heads."""
    def __init__(
        self,
        input_size: int,
        early_layers: list,
        late_layers: list,
        lstm_layers: int = 2,
        lstm_hidden_size: int = 512,
        num_actions: int = MDBO.action_space(),
        num_teampreview_actions: int = MDBO.teampreview_space(),
        max_seq_len: int = 40,
        dropout: float = 0.1,
        gated_residuals: bool = False,
        early_attention_heads: int = 8,
        late_attention_heads: int = 8,
        use_grouped_encoder: bool = False,
        group_sizes=None,
        grouped_encoder_hidden_dim: int = 128,
        grouped_encoder_aggregated_dim: int = 1024,
        pokemon_attention_heads: int = 2,
        teampreview_head_layers: Optional[list] = None,
        teampreview_head_dropout: float = 0.1,
        teampreview_attention_heads: int = 4,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.use_grouped_encoder = use_grouped_encoder
        self.hidden_size = early_layers[-1] if early_layers else input_size
        self.num_actions = num_actions
        self.num_teampreview_actions = num_teampreview_actions
        self.early_attention_heads = early_attention_heads
        self.late_attention_heads = late_attention_heads
        self.teampreview_head_layers = teampreview_head_layers or []

        ResBlock = ResidualBlock

        if use_grouped_encoder and group_sizes is not None:
            self.feature_encoder: Optional[GroupedFeatureEncoder] = GroupedFeatureEncoder(
                group_sizes=group_sizes,
                hidden_dim=grouped_encoder_hidden_dim,
                aggregated_dim=grouped_encoder_aggregated_dim,
                dropout=dropout,
                pokemon_attention_heads=pokemon_attention_heads
            )
            early_ff_layers = []
            prev_size = grouped_encoder_aggregated_dim
            for h in early_layers:
                early_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.early_ff_stack = (
                torch.nn.Sequential(*early_ff_layers)
                if early_ff_layers
                else torch.nn.Identity()
            )
        else:
            self.feature_encoder = None
            input_proj = torch.nn.Linear(input_size, early_layers[0])
            torch.nn.init.kaiming_normal_(input_proj.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(input_proj.bias, 0)
            self.input_proj = input_proj

            early_ff_layers = []
            prev_size = early_layers[0]
            for h in early_layers[1:]:
                early_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.early_ff_stack = (
                torch.nn.Sequential(*early_ff_layers)
                if early_ff_layers
                else torch.nn.Identity()
            )

        # Teampreview head
        teampreview_ff_layers = []
        prev_size = self.hidden_size
        for h in self.teampreview_head_layers:
            teampreview_ff_layers.append(ResBlock(prev_size, h, dropout=teampreview_head_dropout))
            prev_size = h

        teampreview_output_size = self.teampreview_head_layers[-1] if self.teampreview_head_layers else self.hidden_size

        if teampreview_attention_heads > 0:
            self.teampreview_attn: Optional[torch.nn.MultiheadAttention] = torch.nn.MultiheadAttention(
                teampreview_output_size, teampreview_attention_heads, batch_first=True, dropout=teampreview_head_dropout
            )
            self.teampreview_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(teampreview_output_size)
        else:
            self.teampreview_attn = None
            self.teampreview_ln = None

        self.teampreview_ff_stack = (
            torch.nn.Sequential(*teampreview_ff_layers)
            if teampreview_ff_layers
            else torch.nn.Identity()
        )

        self.teampreview_head = torch.nn.Linear(teampreview_output_size, num_teampreview_actions)
        torch.nn.init.xavier_normal_(self.teampreview_head.weight, gain=0.01)
        torch.nn.init.constant_(self.teampreview_head.bias, 0)

        # Early attention
        if early_attention_heads > 0:
            self.early_attn: Optional[torch.nn.MultiheadAttention] = torch.nn.MultiheadAttention(
                self.hidden_size, early_attention_heads, batch_first=True, dropout=dropout
            )
            self.early_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(self.hidden_size)
        else:
            self.early_attn = None
            self.early_ln = None

        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        lstm_output_size = lstm_hidden_size * 2
        if self.hidden_size != lstm_output_size:
            self.skip_proj: Optional[torch.nn.Linear] = torch.nn.Linear(self.hidden_size, lstm_output_size)
            torch.nn.init.xavier_normal_(self.skip_proj.weight, gain=0.01)
            torch.nn.init.constant_(self.skip_proj.bias, 0)
        else:
            self.skip_proj = None

        # Late attention
        if late_attention_heads > 0:
            self.late_attn: Optional[torch.nn.MultiheadAttention] = torch.nn.MultiheadAttention(
                lstm_output_size, late_attention_heads, batch_first=True, dropout=dropout
            )
            self.late_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(lstm_output_size)
        else:
            self.late_attn = None
            self.late_ln = None

        late_ff_layers = []
        prev_size = lstm_output_size
        for h in late_layers:
            late_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.late_ff_stack = (
            torch.nn.Sequential(*late_ff_layers) if late_ff_layers else torch.nn.Identity()
        )

        output_size = late_layers[-1] if late_layers else lstm_output_size

        self.turn_action_head = torch.nn.Linear(output_size, num_actions)
        torch.nn.init.xavier_normal_(self.turn_action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.turn_action_head.bias, 0)

        win_linear1 = torch.nn.Linear(output_size, 128)
        win_linear2 = torch.nn.Linear(128, 1)

        torch.nn.init.xavier_normal_(win_linear1.weight, gain=0.01)
        torch.nn.init.constant_(win_linear1.bias, 0)
        torch.nn.init.xavier_normal_(win_linear2.weight, gain=0.01)
        torch.nn.init.constant_(win_linear2.bias, 0)

        self.win_head = torch.nn.Sequential(
            win_linear1,
            torch.nn.LayerNorm(128),
            torch.nn.Dropout(dropout),
            win_linear2,
            torch.nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        if self.feature_encoder is not None:
            ff_out_early = self.feature_encoder(x)
        else:
            ff_out_early = self.input_proj(x)

        ff_out_early = self.early_ff_stack(ff_out_early)

        # Teampreview head (branches before LSTM)
        teampreview_features = self.teampreview_ff_stack(ff_out_early)

        if self.teampreview_attn is not None:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            tp_attn_out, _ = self.teampreview_attn(
                teampreview_features, teampreview_features, teampreview_features,
                key_padding_mask=attn_mask
            )
            teampreview_features = self.teampreview_ln(teampreview_features + tp_attn_out)  # type: ignore

        teampreview_logits = self.teampreview_head(teampreview_features)

        # Early attention
        if self.early_attn is not None:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            attn_out, _ = self.early_attn(ff_out_early, ff_out_early, ff_out_early, key_padding_mask=attn_mask)
            ff_out_early = self.early_ln(ff_out_early + attn_out)  # type: ignore

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        if positions.max() >= self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x_pos = ff_out_early + self.pos_embedding(positions)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_pos, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )

        if self.skip_proj is not None:
            ff_out_early_proj = self.skip_proj(ff_out_early)
            lstm_out = lstm_out + ff_out_early_proj
        else:
            lstm_out = lstm_out + ff_out_early

        # Late attention
        if self.late_attn is not None:
            attn_mask = ~mask.bool() if mask is not None else None
            attn_out, _ = self.late_attn(
                lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
            )
            out = self.late_ln(lstm_out + attn_out)  # type: ignore
        else:
            out = lstm_out

        out = self.late_ff_stack(out)

        turn_action_logits = self.turn_action_head(out)
        win_logits = self.win_head(out).squeeze(-1)

        return turn_action_logits, teampreview_logits, win_logits


def load_model(model_path: str, config: Dict, device: str = "cpu") -> FlexibleThreeHeadedModel:
    """Load a three-headed model from checkpoint."""
    # Determine input size from config or use embedder to figure it out
    embedder = Embedder(
        format=config.get("battle_format", "gen9vgc2023regulationc"),
        feature_set=Embedder.FULL,
        omniscient=False
    )

    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config.get("lstm_layers", 2),
        lstm_hidden_size=config.get("lstm_hidden_size", 512),
        dropout=config.get("dropout", 0.1),
        gated_residuals=config.get("gated_residuals", False),
        early_attention_heads=config.get("early_attention_heads", 8),
        late_attention_heads=config.get("late_attention_heads", 8),
        use_grouped_encoder=config.get("use_grouped_encoder", False),
        group_sizes=embedder.group_embedding_sizes if config.get("use_grouped_encoder", False) else None,
        grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
        pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=config.get("teampreview_head_layers", []),
        teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=17,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_valid_actions(battle: DoubleBattle) -> List[MDBO]:
    """Get all valid actions for current battle state."""
    valid_actions = []

    if battle.teampreview:
        for i in range(MDBO.teampreview_space()):
            valid_actions.append(MDBO.from_int(i, MDBO.TEAMPREVIEW))
    else:
        for i in range(MDBO.action_space()):
            try:
                mdbo = MDBO.from_int(i, MDBO.TURN)
                dbo = mdbo.to_double_battle_order(battle)
                if is_valid_order(dbo, battle):  # type: ignore
                    valid_actions.append(mdbo)
            except Exception:
                continue

    return valid_actions


def predict_with_model(
    model: FlexibleThreeHeadedModel,
    embedder: Embedder,
    trajectory: List[List[float]],
    battle: DoubleBattle,
    device: str = "cpu"
) -> Tuple[Dict[MDBO, float], float]:
    """
    Get model predictions for current battle state.

    Returns:
        (action_probs, win_prediction)
    """
    # Prepare trajectory tensor
    traj = torch.tensor(trajectory, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, features)
    traj = traj[:, -model.max_seq_len:, :]  # Truncate to max seq length

    model.eval()
    with torch.no_grad():
        turn_action_logits, teampreview_logits, win_logits = model(traj)

        # Get last step predictions
        turn_logits_last = turn_action_logits[0, -1]  # (2025,)
        teampreview_logits_last = teampreview_logits[0, -1]  # (90,)
        win_pred = float(win_logits[0, -1].item())  # scalar

        # Use appropriate head based on battle phase
        if battle.teampreview:
            logits = teampreview_logits_last[:MDBO.teampreview_space()]
            action_type = MDBO.TEAMPREVIEW
            max_actions = MDBO.teampreview_space()
        else:
            logits = turn_logits_last
            action_type = MDBO.TURN
            max_actions = MDBO.action_space()

        # Mask invalid actions
        mask = torch.zeros(logits.size(0), dtype=torch.bool, device=device)
        valid_actions = get_valid_actions(battle)
        valid_indices = [mdbo.to_int() for mdbo in valid_actions]
        mask[valid_indices] = True

        masked_logits = logits.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)

        # Build output dict
        action_probs = {}
        for idx in valid_indices:
            if idx < max_actions:
                action_probs[MDBO.from_int(idx, action_type)] = float(probs[idx].item())

    return action_probs, win_pred


def format_action(mdbo: MDBO, battle: AbstractBattle) -> str:
    """Format action as human-readable string."""
    if battle.teampreview:
        team_order = MDBO.from_int(mdbo.to_int(), MDBO.TEAMPREVIEW)
        return f"Team: {team_order}"
    else:
        try:
            dbo = mdbo.to_double_battle_order(battle)  # type: ignore
            return str(dbo)
        except Exception as e:
            return f"MDBO({mdbo.to_int()}): {e}"


def analyze_battle(
    battle_file: str,
    model: FlexibleThreeHeadedModel,
    embedder: Embedder,
    perspective: str = "p1",
    verbose: bool = False,
    device: str = "cpu"
):
    """Replay battle and analyze model predictions at each step."""
    # Load battle data
    with open(battle_file, 'r') as f:
        battle_json = json.load(f)

    battle_data = BattleData.from_showdown_json(battle_json)

    print(f"\n{'=' * 80}")
    print(f"Battle: {battle_data.battle_id}")
    print(f"Format: {battle_data.format}")
    print(f"Winner: {battle_data.winner}")
    print(f"Perspective: {perspective}")
    print(f"{'=' * 80}\n")

    # Create iterator
    iterator = BattleIterator(battle_data, perspective=perspective, omniscient=False)

    trajectory: List[List[float]] = []
    turn_count = 0
    good_predictions = 0
    bad_predictions = 0

    try:
        while iterator.next_input():

            # Embed state and add to trajectory
            state_vec = embedder.feature_dict_to_vector(embedder.embed(iterator.battle))  # type: ignore
            trajectory.append(state_vec)

            # Get ground truth action
            actual_action = iterator.last_order()

            # Get model predictions
            action_probs, win_pred = predict_with_model(model, embedder, trajectory, iterator.battle, device)  # type: ignore

            # Sort by probability
            sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_actions[:5]

            # Find actual action rank
            actual_rank = None
            for rank, (mdbo, prob) in enumerate(sorted_actions, 1):
                if mdbo == actual_action:
                    actual_rank = rank
                    break

            is_good_pred = actual_rank is not None and actual_rank <= 5
            if is_good_pred:
                good_predictions += 1
            else:
                bad_predictions += 1

            # Display turn header
            turn_type = "TEAMPREVIEW" if iterator.battle.teampreview else f"TURN {turn_count}"
            pred_quality = "✓ GOOD" if is_good_pred else "✗ BAD"

            print(f"\n{'─' * 80}")
            print(f"{turn_type} | Win Prediction: {win_pred:+.3f} | {pred_quality}")
            print(f"{'─' * 80}")

            # Verbose: show battle state
            if verbose:
                print("\n[Battle State]")
                print(battle_to_str(iterator.battle))
                print()

            # Show top-5 predictions
            print("[Top-5 Predictions]")
            for rank, (mdbo, prob) in enumerate(top5, 1):
                action_str = format_action(mdbo, iterator.battle)
                marker = "★" if mdbo == actual_action else " "
                print(f"  {rank}. {marker} {prob:6.2%} | {action_str}")

            # Show actual action if not in top-5
            if actual_rank is None or actual_rank > 5:
                actual_str = format_action(actual_action, iterator.battle)
                actual_prob = action_probs.get(actual_action, 0.0)
                print(f"\n  [Actual Action - Rank {actual_rank or '>1000'}]")
                print(f"  ★ {actual_prob:6.2%} | {actual_str}")
            else:
                print(f"\n  Actual action: Rank {actual_rank}")

            if not iterator.battle.teampreview:
                turn_count += 1

    except StopIteration:
        pass

    # Summary
    print(f"\n{'=' * 80}")
    print("Battle Complete")
    print(f"{'=' * 80}")
    print(f"Good Predictions (actual in top-5): {good_predictions}/{good_predictions + bad_predictions} ({100 * good_predictions / (good_predictions + bad_predictions):.1f}%)")
    print(f"Bad Predictions (actual not in top-5): {bad_predictions}/{good_predictions + bad_predictions} ({100 * bad_predictions / (good_predictions + bad_predictions):.1f}%)")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Replay battles and analyze model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--battle-file",
        type=str,
        required=True,
        help="Path to raw Showdown JSON battle file"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="data/models/three_headed_model.pt",
        help="Path to model checkpoint (.pt file)"
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default="data/models/three_headed_config.json",
        help="Path to model config JSON file"
    )

    parser.add_argument(
        "--perspective",
        type=str,
        choices=["p1", "p2"],
        default="p1",
        help="Which player's perspective to analyze"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full battle state at each turn"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, config, device=args.device)

    # Create embedder
    embedder = Embedder(
        format=config.get("battle_format", "gen9vgc2023regulationc"),
        feature_set=Embedder.FULL,
        omniscient=False
    )

    # Analyze battle
    analyze_battle(
        args.battle_file,
        model,
        embedder,
        perspective=args.perspective,
        verbose=args.verbose,
        device=args.device
    )


if __name__ == "__main__":
    main()
