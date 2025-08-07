# -*- coding: utf-8 -*-
"""
action_prediction_model.py

This script loads a trained action prediction model for Pokémon VGC double battles,
evaluates its predictions on a validation dataset, and demonstrates predictions on a custom battle state.

Key features:
- Defines a DNN model with residual blocks for action prediction.
- Loads and preprocesses battle data using an Embedder and PreprocessedBattleDataset.
- Evaluates model accuracy (top-1, top-3, top-5) across various battle contexts.
- Demonstrates model predictions for a hand-crafted battle scenario.
- Uses action masking to ensure only valid actions are considered.

Usage:
    python src/elitefurretai/scripts/analyze/action_prediction_model.py <val_data_path> <model_path>
"""

import logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from poke_env.battle import DoubleBattle, Move, Pokemon, PokemonType

from elitefurretai.model_utils import MDBO, Embedder, PreprocessedBattleDataset
from elitefurretai.utils.battle_order_validator import is_valid_order


class ResidualBlock(torch.nn.Module):
    """
    Residual block with linear, batch norm, ReLU, and dropout.
    Used as a building block for the DNN model.
    """

    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Linear(in_features, out_features),
                torch.nn.BatchNorm1d(out_features),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.relu(x + residual)  # Add ReLU after addition


class DNN(torch.nn.Module):
    """
    Deep neural network for action prediction, using stacked residual blocks.
    """

    def __init__(self, input_size, hidden_sizes=[1024, 512, 256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_size = input_size

        # Build residual blocks
        for size in hidden_sizes:
            layers.append(ResidualBlock(prev_size, size, dropout))
            prev_size = size

        self.backbone = torch.nn.Sequential(*layers)
        self.action_head = torch.nn.Linear(prev_size, MDBO.action_space())

        # Initialize weights for stability
        for layer in self.backbone:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        torch.nn.init.xavier_normal_(self.action_head.weight)

    def forward(self, x):
        # Forward pass through the residual backbone and output head
        x = self.backbone(x)
        action_logits = self.action_head(x)
        return action_logits


def create_battle(
    p1a: Pokemon = Pokemon(9, species="furret", name="?"),
    p2a: Pokemon = Pokemon(9, species="furret", name="??"),
    p1b: Pokemon = Pokemon(9, species="furret", name="???"),
    p2b: Pokemon = Pokemon(9, species="furret", name="????"),
    doubles: bool = True,
):
    """
    Utility to create a DoubleBattle object with specified Pokémon.
    Used for demonstration and custom scenario prediction.
    """
    if doubles:
        battle = DoubleBattle("tag", "elitefurretai", logging.Logger("example"), gen=9)

        battle._format = "gen9vgc2023regulationc"
        battle.player_role = "p1"

        battle._team = {f"p1: {p1a.name}": p1a, f"p1: {p1b.name}": p1b}
        battle._opponent_team = {f"p2: {p2a.name}": p2a, f"p2: {p2b.name}": p2b}

        # Simulate switching in Pokémon to set up the battle state
        for position, mon in zip(["p1a", "p1b", "p2a", "p2b"], [p1a, p1b, p2a, p2b]):
            battle.switch(
                f"{position}: {mon.name}",
                f"{mon.species}, L{mon.level}",
                f"{mon.current_hp}/{mon.max_hp}",
            )

        return battle


@torch.no_grad()
def analyze(model, dataloader, feature_names):
    """
    Evaluates the model on a validation dataloader and prints accuracy metrics
    (top-1, top-3, top-5) across various battle contexts (turn, action type, KO, etc.).
    """
    model.eval()
    eval_metrics = {"total_top1": 0, "total_top3": 0, "total_top5": 0, "total_samples": 0}
    turns: Dict[int, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())
    ko_can_be_taken: Dict[bool, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())
    mons_alive: Dict[int, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())
    action_type: Dict[str, Dict[str, int]] = defaultdict(lambda: eval_metrics.copy())

    feature_names = {name: i for i, name in enumerate(feature_names)}
    ko_features = {v for k, v in feature_names.items() if "KO" in k}

    print("Starting analysis...")
    for batch in dataloader:
        states, actions, action_masks, _, masks = batch
        states = states.to("cpu")
        actions = actions.to("cpu")
        action_masks = action_masks.to("cpu")
        masks = masks.to("cpu")

        valid_mask = masks.bool()
        if valid_mask.sum() == 0:
            continue

        # Forward pass through the model
        action_logits = model(states[valid_mask])

        # Mask out invalid actions by setting their logits to -inf
        masked_action_logits = action_logits.clone()
        masked_action_logits[~action_masks[valid_mask].bool()] = float("-inf")

        # Filter out samples with no valid actions
        valid_action_rows = action_masks[valid_mask].sum(dim=1) > 0
        if valid_action_rows.sum() == 0 or masked_action_logits.size(0) == 0:
            continue

        masked_action_logits = masked_action_logits[valid_action_rows]
        actions_for_loss = actions[valid_mask][valid_action_rows]

        # Action predictions: top-1, top-3, top-5
        top1_preds = torch.argmax(masked_action_logits, dim=1)
        top3_preds = torch.topk(masked_action_logits, k=3, dim=1).indices
        top5_preds = torch.topk(masked_action_logits, k=5, dim=1).indices

        # Check correctness for each prediction type
        top1_correct = top1_preds == actions_for_loss
        top3_correct = (actions_for_loss.unsqueeze(1) == top3_preds).any(dim=1)
        top5_correct = (actions_for_loss.unsqueeze(1) == top5_preds).any(dim=1)

        for i, state in enumerate(states[valid_mask]):
            # Generate keys for analysis (elo, turn, action type, etc.)
            turn = state[feature_names["turn"]].item()
            turn_type = ""
            action_msg = MDBO.from_int(int(actions[valid_mask][i]), MDBO.TURN).message
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

            # Aggregate metrics for each context
            for key, value in zip(
                ["total_top1", "total_top3", "total_top5", "total_samples"],
                [
                    top1_correct[i].item(),
                    top3_correct[i].item(),
                    top5_correct[i].item(),
                    1,
                ],
            ):
                ko_can_be_taken[can_ko][key] += value
                mons_alive[num_alive][key] += value
                turns[turn][key] += value
                action_type[turn_type][key] += value

    # Print accuracy results for each context
    print("Analysis complete! Results:")
    data: List[Tuple[Dict[Any, Dict[str, int]], str]] = [
        (turns, "Turn"),
        (action_type, "Action Types"),
        (ko_can_be_taken, "Can KO"),
        (mons_alive, "Mon Count"),
    ]
    for results, name in data:
        print(f"\nAnalysis by {name}:")
        for key in sorted(list(results.keys())):
            metrics = results[key]
            if metrics["total_samples"] > 0:
                top1_acc = metrics["total_top1"] / metrics["total_samples"] * 100
                top3_acc = metrics["total_top3"] / metrics["total_samples"] * 100
                top5_acc = metrics["total_top5"] / metrics["total_samples"] * 100
                print(
                    f"  {key}: Top-1: {top1_acc:.1f}%, Top-3: {top3_acc:.1f}%, Top-5: {top5_acc:.1f}% | Samples: {metrics['total_samples']}"
                )
            else:
                print(f"\t{key}: No samples")


def generate_predictions(model, embedder, battle, verbose=False):
    """
    Given a model, embedder, and battle state, generate and print the probability
    distribution over all valid actions for the current state.
    """
    # 1. Embed the current battle state
    state = torch.tensor(
        embedder.feature_dict_to_vector(embedder.embed(battle)), dtype=torch.float32
    ).unsqueeze(0)

    # 2. Generate action mask for the current state
    action_mask = torch.zeros(MDBO.action_space(), dtype=torch.bool)
    input_type = MDBO.TEAMPREVIEW if battle.teampreview else MDBO.TURN
    for possible_action_int in range(MDBO.action_space()):
        try:
            if input_type == MDBO.TEAMPREVIEW:
                if possible_action_int < MDBO.teampreview_space():
                    action_mask[possible_action_int] = 1
            else:
                dbo = MDBO.from_int(
                    possible_action_int, input_type
                ).to_double_battle_order(battle)
                if is_valid_order(dbo, battle):
                    action_mask[possible_action_int] = 1
        except Exception:
            continue

    # 3. Model prediction
    model.eval()
    with torch.no_grad():
        logits = model(state)
        logits[0, ~action_mask] = float("-inf")  # Mask out invalid actions
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_actions]

    # 4. Collect predictions for all valid actions
    preds = {}
    for idx in torch.where(action_mask)[0]:
        try:
            dbo = MDBO.from_int(idx.item(), input_type).to_double_battle_order(battle)
            if is_valid_order(dbo, battle):
                preds[dbo.message] = probs[idx].item()
        except Exception:
            continue

    if verbose:
        print("Predictions:")
        for action, prob in sorted(preds.items(), key=lambda v: v[1], reverse=True):
            print(f"{action}: {(100 * prob):.2f}%")

    return preds


# Main entry point: analyze predictions at scale, then demonstrate on a custom battle state
def main(val_dir_path, model_filepath):
    """
    Loads the model and validation data, runs large-scale evaluation, and demonstrates
    predictions on a custom battle scenario.
    """
    # Prepare embedder and dataset
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=True
    )
    dataset = PreprocessedBattleDataset(
        val_dir_path,
        embedder=embedder,
    )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=4, pin_memory=True
    )

    # Initialize and load the trained model
    model = DNN(embedder.embedding_size)
    model.load_state_dict(torch.load(model_filepath))
    model.eval()

    # Run large-scale analysis on the validation set
    analyze(model, loader, embedder.feature_names)

    print("Done with analysis, now creating a custom state!")

    # Create a custom battle state for demonstration
    rilla = Pokemon(9, species="Rillaboom")
    rilla.stats = {"hp": 207, "atk": 165, "def": 111, "spa": 72, "spd": 116, "spe": 107}
    rilla.item, rilla.ability, rilla._terastallized_type = (
        "assaultvest",
        "grassysurge",
        PokemonType.FIRE,
    )
    rilla._moves = {
        m: Move(m, 9) for m in ["fakeout", "grassyglide", "woodhammer", "uturn"]
    }
    rilla.set_hp("207/207")

    csr = Pokemon(9, species="Calyrex-Shadow")
    csr.stats = {"hp": 176, "atk": 94, "def": 100, "spa": 217, "spd": 120, "spe": 222}
    csr.item, csr.ability, csr._terastallized_type = "lifeorb", "asone", PokemonType.DARK
    csr._moves = {
        m: Move(m, 9) for m in ["astralbarrage", "psychic", "protect", "nastyplot"]
    }
    csr.boosts = {"spa": 1}
    csr.set_hp(str(int(176 * 0.3)) + "/176")

    smeargle = Pokemon(9, species="Smeargle")
    smeargle.stats = {"hp": 162, "atk": 41, "def": 55, "spa": 36, "spd": 65, "spe": 139}
    smeargle.item, smeargle.ability, smeargle._terastallized_type = (
        "focussash",
        "moody",
        PokemonType.GHOST,
    )
    smeargle._moves = {m: Move(m, 9) for m in ["fakeout", "spore", "followme", "decorate"]}
    smeargle.set_hp(str(1) + "/162")

    cir = Pokemon(9, species="Calyrex-Ice")
    cir.stats = {"hp": 252, "atk": 231, "def": 170, "spa": 105, "spd": 158, "spe": 63}
    cir.item, cir.ability, cir._terastallized_type = (
        "assaultvest",
        "asone",
        PokemonType.GRASS,
    )
    cir._moves = {
        m: Move(m, 9) for m in ["glaciallance", "seedbomb", "highhorsepower", "crunch"]
    }
    cir.boosts = {"atk": 1}
    cir.set_hp(str(int(252 * 0.7)) + "/252")
    cir.terastallize("Grass")

    # Build the battle state with the above Pokémon
    battle = create_battle(
        p1a=csr,
        p2a=cir,
        p1b=rilla,
        p2b=smeargle,
    )
    battle.field_start("Grassy Terrain")
    battle._available_moves = [list(csr.moves.values()), list(rilla.moves.values())]
    battle._can_tera = [True, True]

    # Generate and print predictions for the custom battle state
    generate_predictions(model, embedder, battle, verbose=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
