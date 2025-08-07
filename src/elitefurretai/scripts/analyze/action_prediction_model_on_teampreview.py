# -*- coding: utf-8 -*-
"""
analyze_teampreview_model.py

This script loads a trained team preview model and evaluates its predictions on two example VGC teams.
It builds a DoubleBattle object with two teams, runs the model to predict the probability of each possible
team preview selection (which 4 Pokémon to bring), and prints the probabilities for each possible selection,
including which Pokémon are in the lead and back positions.

Key features:
- Loads a trained model from a .pth file.
- Constructs two example teams using Showdown format strings.
- Builds a DoubleBattle object for team preview.
- Uses the model to predict the probability distribution over all possible team preview choices.
- Prints the most likely team preview choices and their probabilities, including which Pokémon are selected.

Usage:
    python src/elitefurretai/scripts/analyze/analyze_teampreview_model.py path/to/model.pth
"""

import logging
import sys

import torch
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from elitefurretai.model_utils import MDBO, Embedder

# Example VGC teams in Showdown format
TEAM1 = """
Dozo (Palafin-Hero) @ Mystic Water
Ability: Zero to Hero
Level: 50
Tera Type: Water
EVs: 228 HP / 252 Atk / 4 Def / 4 SpD / 20 Spe
Adamant Nature
- Haze
- Wave Crash
- Jet Punch
- Protect

Bundle (Chien-Pao) @ Focus Sash
Ability: Sword of Ruin
Level: 50
Tera Type: Ghost
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Sacred Sword
- Ice Spinner
- Sucker Punch
- Protect

Dnite (Dragapult) @ Choice Band
Ability: Clear Body
Level: 50
Tera Type: Steel
EVs: 36 HP / 252 Atk / 4 Def / 4 SpD / 212 Spe
Jolly Nature
- Dragon Darts
- Phantom Force
- Tera Blast
- U-turn

WoChien (Amoonguss) @ Sitrus Berry
Ability: Regenerator
Level: 50
Tera Type: Water
EVs: 244 HP / 100 Def / 164 SpD
Relaxed Nature
IVs: 0 Atk / 0 Spe
- Spore
- Rage Powder
- Pollen Puff
- Protect

ChiYu (Arcanine) @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Flying
EVs: 236 HP / 68 Atk / 4 Def / 4 SpD / 196 Spe
Jolly Nature
- Will-O-Wisp
- Flare Blitz
- Extreme Speed
- Protect

Mimi (Flutter Mane) @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 100 HP / 244 Def / 44 SpA / 4 SpD / 116 Spe
Modest Nature
IVs: 0 Atk
- Moonblast
- Dazzling Gleam
- Shadow Ball
- Protect
"""

TEAM2 = """
Gothitelle (F) @ Sitrus Berry
Ability: Shadow Tag
Level: 50
Tera Type: Water
EVs: 252 HP / 100 Def / 156 SpD
Calm Nature
- Trick Room
- Fake Out
- Psychic
- Helping Hand

Iron Hands @ Safety Goggles
Ability: Quark Drive
Level: 50
Tera Type: Fire
EVs: 196 HP / 76 Atk / 4 Def / 228 SpD / 4 Spe
Adamant Nature
- Swords Dance
- Drain Punch
- Wild Charge
- Protect

Flutter Mane @ Life Orb
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Dazzling Gleam
- Moonblast
- Shadow Ball
- Protect

Chien-Pao @ Focus Sash
Ability: Sword of Ruin
Level: 50
Tera Type: Dark
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Protect
- Ice Spinner
- Throat Chop
- Sacred Sword

Palafin @ Mystic Water
Ability: Zero to Hero
Level: 50
Tera Type: Water
EVs: 76 HP / 252 Atk / 4 Def / 172 SpD / 4 Spe
Adamant Nature
- Jet Punch
- Wave Crash
- Haze
- Protect

Amoonguss @ Wiki Berry
Ability: Regenerator
Tera Type: Water
EVs: 252 HP / 100 Def / 156 SpD
Sassy Nature
IVs: 0 Atk / 0 Spe
- Pollen Puff
- Spore
- Protect
- Rage Powder
"""


# Model definition. Note that to load the model, you'll need to have the right
# definition of the model here
class ResidualBlock(torch.nn.Module):
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
    def __init__(
        self,
        hidden_sizes=[1024, 512, 256, 128],
        dropout=0.3,
        embedder=Embedder(
            format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=True
        ),
    ):
        super().__init__()
        self.embedder = embedder
        layers = []
        prev_size = embedder.embedding_size

        # Build residual blocks for feature extraction
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

    def predict(self, battle: DoubleBattle):
        """
        Given a DoubleBattle object in teampreview, returns a dictionary mapping
        team preview choices to their predicted probabilities.
        """
        assert battle.teampreview
        self.eval()
        with torch.no_grad():
            # Embed the battle state
            state = self.embedder.feature_dict_to_vector(self.embedder.embed(battle))
            state = torch.tensor(
                state, dtype=torch.float32, device=next(self.parameters()).device
            ).unsqueeze(0)
            logits = self(state)  # shape: (1, num_actions)
            logits = logits.squeeze(0)  # shape: (num_actions,)

            # Mask out actions not in teampreview space by setting logits to -inf
            mask = (
                torch.arange(logits.size(0), device=logits.device)
                < MDBO.teampreview_space()
            )
            masked_logits = logits.masked_fill(~mask, float("-inf"))

            # Apply softmax only to valid actions
            probs = torch.softmax(masked_logits, dim=-1)

        # Return a dict mapping action string to probability for all nonzero actions
        return {
            MDBO.from_int(i, type=MDBO.TEAMPREVIEW).message: prob
            for i, prob in enumerate(probs.cpu().numpy())
            if prob > 0
        }


def main(model_path):
    # Load the trained model from disk
    model = DNN()
    model.load_state_dict(torch.load(model_path))

    # Build a DoubleBattle object for team preview with two example teams
    battle = DoubleBattle("tag", "elitefurretai", logging.Logger("example"), gen=9)
    battle._format = "gen9vgc2023regulationc"
    battle.player_role = "p1"
    battle._teampreview = True

    # Set the player's team for teampreview
    battle.teampreview_team = [
        Pokemon(gen=9, teambuilder=tb_mon) for tb_mon in ConstantTeambuilder(TEAM1).team
    ]

    # Set the opponent's team for teampreview
    battle._teampreview_opponent_team = [
        Pokemon(gen=9, teambuilder=tb_mon) for tb_mon in ConstantTeambuilder(TEAM2).team
    ]

    # Run the model to get predicted probabilities for each team preview choice
    p1_probabilities = model.predict(battle)
    print("P1 probabilities:")
    # Print the most likely team preview choices and their probabilities
    for action, prob in sorted(p1_probabilities.items(), key=lambda v: v[1], reverse=True):
        if prob > 0.01:
            # Parse the action string to get indices of selected Pokémon
            mon1 = battle.teampreview_team[int(action[-4]) - 1]
            mon2 = battle.teampreview_team[int(action[-3]) - 1]
            mon3 = battle.teampreview_team[int(action[-2]) - 1]
            mon4 = battle.teampreview_team[int(action[-1]) - 1]
            print(
                f"{action}: {(prob * 100):.2f}% || Lead [{mon1.species}, {mon2.species}], Back [{mon3.species}, {mon4.species}]"
            )


# Usage: python src/elitefurretai/scripts/analyze/analyze_teampreview_model.py path/to/model.pth
if __name__ == "__main__":
    main(sys.argv[1])
