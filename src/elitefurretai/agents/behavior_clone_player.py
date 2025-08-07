import copy
from typing import Dict, List

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.model_double_battle_order import MDBO
from elitefurretai.utils.battle_order_validator import is_valid_order


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
        self.max_seq_len: int = max_seq_len
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


class BehaviorClonePlayer(Player):
    def __init__(
        self,
        model_filepath: str,
        battle_format: str = "gen9vgc2023regulationc",
        probabilistic=True,
        **kwargs,
    ):
        # pull in all player manually
        super().__init__(**kwargs, battle_format=battle_format)
        self._embedder = Embedder(
            format=battle_format, feature_set=Embedder.FULL, omniscient=False
        )
        self._probabilistic = probabilistic
        self._trajectories: Dict[str, list] = {}

        # The model that we use to make predictions
        self.model: TwoHeadedHybridModel = TwoHeadedHybridModel(
            self._embedder.embedding_size
        )
        self.load_model(model_filepath)
        self.model.eval()

        self._last_message_error: Dict[str, bool] = {}
        self._last_message: Dict[str, str] = {}

    async def send_message(self, message: str, room: str):
        self._last_message[room] = message
        await self.ps_client.send_message(room, message)

    # Wrote some basic unnecessary code to dictate whether the last message was an error
    async def handle_battle_message(self, split_messages: List[List[str]]):
        if (
            len(split_messages) > 1
            and len(split_messages[1]) > 1
            and split_messages[1][1] == "init"
        ):
            battle_info = split_messages[0][0].split("-")
            battle = await self._create_battle(battle_info)
        else:
            battle = await self._get_battle(split_messages[0][0])

        if split_messages[0][0] == "error" and split_messages[0][1] in [
            "[Unavailable choice]",
            "[Invalid choice]",
        ]:
            self._last_message_error[battle.battle_tag] = True
        else:
            self._last_message_error[battle.battle_tag] = False
        await super()._handle_battle_message(split_messages)

    def last_message_error(self, room) -> bool:
        return self._last_message_error.get(room, False)

    def last_message(self, room: str) -> str:
        assert room in self._last_message, f"No last message for room {room}"
        return self._last_message[room]

    def reset_battles(self):
        """Reset the battles dictionary to start fresh."""
        self._battles = {}
        self._trajectories = {}

    def load_model(self, filepath: str):
        assert self.model is not None
        self.model.load_state_dict(torch.load(filepath))

    def embed_battle_state(self, battle: AbstractBattle) -> List[float]:
        assert isinstance(battle, DoubleBattle)
        assert self._embedder.embedding_size == len(self._embedder.embed(battle))
        return self._embedder.feature_dict_to_vector(self._embedder.embed(battle))

    def predict(self, traj: torch.Tensor, battle: DoubleBattle) -> Dict[MDBO, float]:
        """
        Given a trajectory tensor and battle, returns a dict of valid actions and their probabilities
        for the last state in the trajectory.
        """
        traj = traj[:, -self.model.max_seq_len :, :]  # type: ignore
        self.model.eval()
        with torch.no_grad():
            # Forward pass: get logits for all steps in the trajectory
            action_logits, _ = self.model(
                traj
            )  # shape: (seq_len, num_actions) or (batch, seq_len, num_actions)
            if action_logits.dim() == 3:
                # Remove batch dimension if present
                action_logits = action_logits.squeeze(0)
            # Always use the last state in the trajectory
            last_logits = action_logits[-1]  # shape: (num_actions,)

            # Build mask for valid actions
            if battle.teampreview:
                mask = (
                    torch.arange(last_logits.size(0), device=last_logits.device)
                    < MDBO.teampreview_space()
                )
            else:
                mask = torch.zeros(
                    last_logits.size(0), dtype=torch.bool, device=last_logits.device
                )
                for i in range(last_logits.size(0)):
                    try:
                        dbo = MDBO.from_int(i, MDBO.TURN).to_double_battle_order(battle)
                        if is_valid_order(dbo, battle):  # type: ignore
                            mask[i] = 1
                    except Exception:
                        continue

            # Mask out invalid actions
            masked_logits = last_logits.masked_fill(~mask, float("-inf"))

            # Softmax over valid actions
            probs = torch.softmax(masked_logits, dim=-1)

            # Build output dict
            if battle.teampreview:
                return {
                    MDBO.from_int(i, type=MDBO.TEAMPREVIEW): float(prob)
                    for i, prob in enumerate(probs.cpu().numpy())
                    if float(prob) > 0 and i < MDBO.teampreview_space()
                }
            else:
                return {
                    MDBO.from_int(i, type=MDBO.TURN): float(prob)
                    for i, prob in enumerate(probs.cpu().numpy())
                    if float(prob) > 0
                }

    """
    PLAYER-BASED METHODS
    """

    @property
    def probabilistic(self):
        return self._probabilistic

    @probabilistic.setter
    def probabilistic(self, value: bool):
        self._probabilistic = value

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        assert isinstance(battle, DoubleBattle)

        # Embed and store the battle state
        state_vec = self.embed_battle_state(battle)
        if battle.battle_tag not in self._trajectories:
            self._trajectories[battle.battle_tag] = []
        self._trajectories[battle.battle_tag].append(state_vec)

        # Get model prediction based on the battle state
        predictions: Dict[MDBO, float] = self.predict(
            torch.Tensor(self._trajectories[battle.battle_tag]).unsqueeze(0), battle
        )
        keys = list(predictions.keys())

        if len(keys) == 0:
            # print("No valid actions available, returning random move.")
            return DefaultBattleOrder()

        probabilities = np.array(list(predictions.values()))
        probabilities = probabilities / probabilities.sum()  # Ensure sum to 1

        # If probabilistic, sample a move proportional to the softmax; otherwise, choose the best move
        if self._probabilistic:
            choice_idx = np.random.choice(len(keys), p=probabilities)
        else:
            choice_idx = int(np.argmax(probabilities))

        chosen_move = keys[choice_idx]
        return chosen_move

    def teampreview(self, battle: AbstractBattle) -> str:
        assert battle.player_role
        message = self.choose_move(battle).message

        # Need to populate team with teampreview mon's stats
        battle.team = {
            mon.identifier(battle.player_role): copy.deepcopy(mon)
            for mon in map(
                lambda x: battle.teampreview_team[int(x) - 1],
                message.replace("/team ", ""),
            )
        }

        return message

    # Save it to the battle_filepath using DataProcessor, using opponent information
    # to create omniscient BattleData object
    def _battle_finished_callback(self, battle: AbstractBattle):
        pass
