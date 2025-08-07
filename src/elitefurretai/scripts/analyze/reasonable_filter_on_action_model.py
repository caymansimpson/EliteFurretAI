# -*- coding: utf-8 -*-
"""
reasonable_filter_on_action_model.py

This script loads a trained action prediction model for Pokémon VGC double battles,
evaluates its predictions on a dataset, and looks at whether filtering it to reasonable moves
after training improves model performance

Key features:
- Defines a DNN model with residual blocks for action prediction.
- Loads and preprocesses battle data using an Embedder and PreprocessedBattleDataset.
- Evaluates model accuracy (top-1, top-3, top-5) across various battle contexts.
- Demonstrates model predictions for a hand-crafted battle scenario.
- Uses action masking to ensure only valid actions are considered.

Usage:
    python src/elitefurretai/scripts/analyze/action_prediction_model.py <val_data_path> <model_path>

Analysis shows that the model already picks these heuristics up, and it only performs slightly better
when filtering to reasonable moves, so this is not a useful heuristic to apply at inference time.
"""

import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import orjson
import torch

from elitefurretai.model_utils import MDBO, BattleIteratorDataset, Embedder
from elitefurretai.utils.battle_order_validator import (
    is_reasonable_move,
    is_valid_order,
)


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
        num_lstm_layers=2,
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


def generate_data(dataloader):
    """
    Iterates through BattleIterator objects from the dataloader and returns:
    - states: (batch, seq_len, input_size)
    - actions: (batch, seq_len)
    - action_masks: (batch, seq_len, num_actions)
    - reasonable_action_masks: (batch, seq_len, num_actions)
    - wins: (batch, seq_len)
    - masks: (batch, seq_len)
    - turns_til_end: (batch, seq_len)  # number of turns left in the battle at each step

    This mimics BattleDataset but adds reasonable_action_masks and turns_til_end.
    """
    batch_states = []
    batch_actions = []
    batch_action_masks = []
    batch_reasonable_action_masks = []
    batch_wins = []
    batch_masks = []
    batch_turns_til_end = []

    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=True
    )
    max_seq_len = 17

    for batch in dataloader:
        # batch is a list of BattleIterator objects (batch_size,)
        for iter in batch:
            bd = iter.bd  # The underlying BattleData object
            seq_len = getattr(bd, "turns", max_seq_len)
            seq_len = min(seq_len, max_seq_len)  # truncate if needed

            # Preallocate
            states = torch.zeros(max_seq_len, embedder.embedding_size)
            actions = torch.zeros(max_seq_len, dtype=torch.long)
            action_masks = torch.zeros(max_seq_len, MDBO.action_space())
            reasonable_action_masks = torch.zeros(max_seq_len, MDBO.action_space())
            wins = torch.zeros(max_seq_len)
            masks = torch.zeros(max_seq_len)
            turns_til_end = torch.zeros(max_seq_len, dtype=torch.long)

            # Reset iterator for this perspective
            i = 0
            while not iter.battle.finished and iter.next_input() and i < seq_len:
                input = iter.last_input
                if input is None:
                    continue

                # Build action mask
                if iter.last_input_type == MDBO.TEAMPREVIEW:
                    for possible_action_int in range(MDBO.action_space()):
                        if possible_action_int < MDBO.teampreview_space():
                            action_masks[i, possible_action_int] = 1
                            reasonable_action_masks[i, possible_action_int] = (
                                1  # All teampreview actions are reasonable
                            )
                else:
                    assert iter.last_input_type is not None
                    for possible_action_int in range(MDBO.action_space()):
                        try:
                            dbo = MDBO.from_int(
                                possible_action_int, iter.last_input_type
                            ).to_double_battle_order(iter.battle)
                            if is_valid_order(dbo, iter.battle):
                                action_masks[i, possible_action_int] = 1
                                reasonable_action_masks[i, possible_action_int] = 1

                                # Reasonable mask: must also be is_reasonable_move
                                if (
                                    iter.last_input_type == MDBO.TURN
                                    and not is_reasonable_move(dbo, iter.battle)
                                ):
                                    reasonable_action_masks[i, possible_action_int] = 0
                        except Exception:
                            continue

                # Get the target action index for this step
                action_idx = iter.last_order().to_int()

                # Embed state
                states[i] = torch.tensor(
                    embedder.feature_dict_to_vector(embedder.embed(iter.battle))
                )
                actions[i] = action_idx
                wins[i] = int(bd.winner == iter.battle.player_username)
                masks[i] = 1

                # Mask out struggle, revival blessing, and invalid actions
                if (
                    "struggle" in iter.last_input
                    or "revivalblessing" in iter.last_input
                    or action_masks[i, action_idx] == 0
                ):
                    masks[i] = 0

                # Turns until end = total turns - current turn (turn is 0-indexed)
                turns_til_end[i] = max(0, getattr(bd, "turns", seq_len) - iter.battle.turn)

                i += 1

            batch_states.append(states)
            batch_actions.append(actions)
            batch_action_masks.append(action_masks)
            batch_reasonable_action_masks.append(reasonable_action_masks)
            batch_wins.append(wins)
            batch_masks.append(masks)
            batch_turns_til_end.append(turns_til_end)

    # Stack into tensors of shape (batch, seq_len, ...)
    states = torch.stack(batch_states, dim=0)
    actions = torch.stack(batch_actions, dim=0)
    action_masks = torch.stack(batch_action_masks, dim=0)
    reasonable_action_masks = torch.stack(batch_reasonable_action_masks, dim=0)
    wins = torch.stack(batch_wins, dim=0)
    masks = torch.stack(batch_masks, dim=0)
    turns_til_end = torch.stack(batch_turns_til_end, dim=0)

    return (
        states,
        actions,
        action_masks,
        reasonable_action_masks,
        wins,
        masks,
        turns_til_end,
    )


def identity_collate(batch):
    return batch


# Main entry point: analyze predictions at scale, then demonstrate on a custom battle state
def main(battlefile_path, model_filepath, num_battles):
    """
    Loads the model and validation data, runs large-scale evaluation, and demonstrates
    predictions on a custom battle scenario.
    """
    # Load files
    files = []
    with open(battlefile_path, "rb") as f:
        files = orjson.loads(f.read())[:num_battles]

    # Prepare embedder and dataset
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=True
    )
    dataset = BattleIteratorDataset(files)

    # Initialize and load the trained model
    model = TwoHeadedHybridModel(embedder.embedding_size)
    model.load_state_dict(torch.load(model_filepath))
    model.eval()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=4,
        collate_fn=identity_collate,  # disables default collation, returns a list of BattleIterator
    )

    (
        states,
        actions,
        action_masks,
        reasonable_action_masks,
        wins,
        masks,
        turns_til_end,
    ) = generate_data(dataloader)

    print("Analysis with regular action masks")
    analyze(
        model,
        states,
        actions,
        action_masks,
        wins,
        masks,
        turns_til_end,
        embedder.feature_names,
    )

    print("Analysis with reasonable action masks")
    analyze(
        model,
        states,
        actions,
        reasonable_action_masks,
        wins,
        masks,
        turns_til_end,
        embedder.feature_names,
    )


def analyze(
    model, states, actions, action_masks, wins, masks, turns_til_end, feature_names
):
    """
    Evaluates the model on the provided tensors (from generate_data) and prints accuracy metrics
    (top-1, top-3, top-5, win) across various battle contexts (turn, action type, KO, etc.).
    """
    model.eval()
    eval_metrics = {
        "total_top1": 0,
        "total_top3": 0,
        "total_top5": 0,
        "win": 0,
        "total_samples": 0,
    }
    turns: Dict[int, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())
    ko_can_be_taken: Dict[bool, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())
    mons_alive: Dict[int, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())
    action_type: Dict[str, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())
    tte: Dict[int, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())

    feature_names = {name: i for i, name in enumerate(feature_names)}
    ko_features = {v for k, v in feature_names.items() if "KO" in k}

    # Forward pass: (batch, seq_len, ...)
    action_logits, win_logits = model(states, masks)

    # Mask invalid actions (set logits for invalid actions to -inf)
    masked_action_logits = action_logits.masked_fill(~action_masks.bool(), float("-inf"))

    # Get predictions
    top1_preds = torch.argmax(masked_action_logits, dim=-1)  # (batch, seq_len)
    top3_preds = torch.topk(
        masked_action_logits, k=3, dim=-1
    ).indices  # (batch, seq_len, 3)
    top5_preds = torch.topk(
        masked_action_logits, k=5, dim=-1
    ).indices  # (batch, seq_len, 5)
    win_preds = (torch.sigmoid(win_logits) > 0.5).float()  # (batch, seq_len)

    # Check correctness for each metric
    top1_correct = top1_preds == actions  # (batch, seq_len)
    top3_correct = (actions.unsqueeze(-1) == top3_preds).any(dim=-1)  # (batch, seq_len)
    top5_correct = (actions.unsqueeze(-1) == top5_preds).any(dim=-1)  # (batch, seq_len)
    win_correct = (win_preds == wins).float()  # (batch, seq_len)

    # Only consider valid (unpadded) positions
    valid_mask = masks.bool()  # (batch, seq_len)
    batch_idx, seq_idx = torch.where(valid_mask)

    for b, t in zip(batch_idx.tolist(), seq_idx.tolist()):
        state = states[b, t]
        action = actions[b, t]
        turn = state[feature_names["turn"]].item()
        num_turns = turns_til_end[b, t].item()
        action_msg = MDBO.from_int(int(action), MDBO.TURN).message
        # Determine turn type
        if int(state[feature_names["teampreview"]].item()) == 1:
            turn_type = "teampreview"
        elif any(
            int(state[feature_names[f"MON:{j}:force_switch"]].item()) == 1
            for j in range(6)
        ):
            turn_type = "force_switch"
        elif "switch" in action_msg and "move" in action_msg:
            turn_type = "both"
        elif "move" in action_msg:
            turn_type = "move"
        elif "switch" in action_msg:
            turn_type = "switch"
        else:
            continue  # Skip if we can't determine the turn type

        can_ko = max(state[feature_idx] for feature_idx in ko_features).item()
        num_alive = int(
            8
            - state[feature_names["OPP_NUM_FAINTED"]].item()
            - state[feature_names["NUM_FAINTED"]].item()
        )

        for key, value in zip(
            ["total_top1", "total_top3", "total_top5", "win", "total_samples"],
            [
                top1_correct[b, t].item(),
                top3_correct[b, t].item(),
                top5_correct[b, t].item(),
                win_correct[b, t].item(),
                1,
            ],
        ):
            ko_can_be_taken[can_ko][key] += value
            mons_alive[num_alive][key] += value
            turns[turn][key] += value
            action_type[turn_type][key] += value
            tte[num_turns][key] += value

    # Print accuracy
    print("\tAnalysis complete! Results:")
    data: List[Tuple[Dict[Any, Dict[str, int]], str]] = [
        (turns, "Turn"),
        (action_type, "Action Types"),
        (ko_can_be_taken, "Can KO"),
        (mons_alive, "Mons Alive"),
    ]
    for results, name in data:
        print(f"\n\tAnalysis by {name}:")
        for key in sorted(list(results.keys())):
            metrics = results[key]
            if metrics["total_samples"] > 0:
                top1_acc = metrics["total_top1"] / metrics["total_samples"] * 100
                top3_acc = metrics["total_top3"] / metrics["total_samples"] * 100
                top5_acc = metrics["total_top5"] / metrics["total_samples"] * 100
                win_acc = metrics["win"] / metrics["total_samples"] * 100
                print(
                    f"\t\t{key}: Top-1: {top1_acc:.1f}%, Top-3: {top3_acc:.1f}%, Top-5: {top5_acc:.1f}%, Win: {win_acc:.1f}% | Samples: {metrics['total_samples']}"
                )
            else:
                print(f"\t\t{key}: No samples found")


# python src/elitefurretai/scripts/analyze/reasonable_filter_on_action_model.py <battle_filepath> <model_path> <num_battles>
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
