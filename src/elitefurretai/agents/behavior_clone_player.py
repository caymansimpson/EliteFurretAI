import copy
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.encoder import MDBO
from elitefurretai.utils.battle_order_validator import is_valid_order

"""
Architecture Overview:
======================

Input: (batch, seq_len, 9223 features)
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│ GROUPED FEATURE ENCODER                                         │
│                                                                 │
│ Split features into semantic groups:                            │
│   • Player Pokemon 0-5    (6 groups)                            │
│   • Opponent Pokemon 0-5  (6 groups)                            │
│   • Battle state          (1 group)                             │
│   • Engineered features   (1 group)                             │
│                                                                 │
│ Each group: Linear(group_size → 128) → LN → ReLU → Dropout      │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ CROSS-GROUP ATTENTION (Pokemon Synergies)                   │ │
│ │                                                             │ │
│ │ Player Pokemon features attend to each other:               │ │
│ │   Pokemon_0 ←→ Pokemon_1 ←→ ... ←→ Pokemon_5                │ │
│ │   (Learns team synergies and interactions)                  │ │
│ │                                                             │ │
│ │ MultiheadAttention(6 heads, 128 dim)                        │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Aggregate: Concatenate all groups → Linear(14*128 = 1792) → 1024│
└─────────────────────────────────────────────────────────────────┘
    |
    v
(batch, seq_len, 1024)
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│ FEEDFORWARD STACK #1 (Pre-LSTM, 3 ResidualBlocks)               │
│   ResidualBlock(1024 → 1024)                                    │
│   ResidualBlock(1024 → 1024)                                    │
│   ResidualBlock(1024 → 512)                                     │
└─────────────────────────────────────────────────────────────────┘
    |
    | ff_out_1: (batch, seq_len, 512) ← SAVE FOR SKIP CONNECTION
    |─────────────────────────────────────┐
    v                                     │
+ Positional Encoding (512-dim)           │
    |                                     │
    v                                     │
┌─────────────────────────────────────┐   │
│ BIDIRECTIONAL LSTM (2 layers)       │   │
│   Input: 512                        │   │
│   Hidden: 512 per direction         │   │
│   Output: 1024 (bidirectional)      │   │
│   (No projection needed)            │   │
└─────────────────────────────────────┘   │
    |                                     │
    | lstm_out: (batch, seq_len, 1024)    │
    |                                     │
    | Project skip connection:            │
    | skip_proj(ff_out_1): 512 → 1024     │
    |                                     │  SKIP CONNECTION
    |<────────────────────────────────────┘ (skip_proj(ff_out_1) + lstm_out)
    v
lstm_out = lstm_out + skip_proj(ff_out_1)
    |
    | (batch, seq_len, 1024)
    v
┌──────────────────────────────────────------------------------───┐
│ FEEDFORWARD STACK #2 (Post-LSTM, 3 ResidualBlocks)              │
│   ResidualBlock(1024 → 1024)                                    │
│   ResidualBlock(1024 → 512)                                     │
│   ResidualBlock(512 → 256)                                      │
└─────────────────────────────────────────────────────────────────┘
    |
    | ff_out_2: (batch, seq_len, 256)
    v
┌─────────────────────────────────────────┐
│ MULTI-HEAD SELF-ATTENTION (8 heads)     │
│   Query, Key, Value: ff_out_2           │
│   Residual: attn_out + ff_out_2         │
│   LayerNorm                             │
└─────────────────────────────────────────┘
    |
    v
out: (batch, seq_len, 256)
    |
    ├─────────────────────────────┐
    v                             v
┌──────────────────┐      ┌─────────────────┐
│ ACTION HEAD      │      │ WIN HEAD        │
│ Linear(256 →     │      │ Linear(256→128) │
│   num_actions)   │      │ LayerNorm       │
└──────────────────┘      │ Linear(128→1)   │
(batch, seq_len,          │ Tanh            │
      action_space)       └─────────────────┘
                           (batch, seq_len)
                           values ∈ [-1, 1]

Key Features:
• Grouped encoding: Separates Pokemon/Battle features for structured learning
• Cross-attention: Pokemon features attend to each other (team synergies)
• Dual FF stacks: Pre-LSTM (1024→1024→512) and Post-LSTM (1024→512→256)
• Hourglass architecture: Wide (1024) → Narrow (512) → Wide (1024) → Narrow (256)
• Larger LSTM: 512-dim hidden states → 1024-dim output (bidirectional)
• Skip connection: FF#1 output (512) projected to 1024, added to LSTM output
• No double ReLU: ResidualBlocks allow negative values for win head
"""


class ResidualBlock(torch.nn.Module):
    """
    Residual block without second ReLU to allow negative values.
    Architecture: Linear → LayerNorm → ReLU → Dropout → Add residual
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features)
        self.ln = torch.nn.LayerNorm(out_features)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

        # Initialize
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
        return x + residual  # No second ReLU - allows negative values


class GatedResidualBlock(torch.nn.Module):
    """
    Gated residual block without second ReLU.
    Architecture: Linear → LN → ReLU → Dropout → Linear → LN → Gate → Add residual
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        # Main path
        self.linear1 = torch.nn.Linear(in_features, out_features)
        self.ln1 = torch.nn.LayerNorm(out_features)
        self.linear2 = torch.nn.Linear(out_features, out_features)
        self.ln2 = torch.nn.LayerNorm(out_features)

        # Gate generation
        gate_linear = torch.nn.Linear(in_features, out_features)
        self.gate = torch.nn.Sequential(
            gate_linear, torch.nn.Sigmoid()
        )

        # Regularization
        self.relu = torch.nn.ReLU()

        # Initialize
        torch.nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.constant_(self.linear2.bias, 0)
        torch.nn.init.xavier_normal_(gate_linear.weight, gain=1.0)
        torch.nn.init.constant_(gate_linear.bias, 0)

        # Projection for residual if dimensions change
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

        # Main path
        main = self.linear1(x)
        main = self.ln1(main)
        main = self.relu(main)
        main = self.linear2(main)
        main = self.ln2(main)

        # Apply gate
        gate_value = self.gate(x)
        gated_output = gate_value * main

        return residual + gated_output  # No final ReLU


class GroupedFeatureEncoder(torch.nn.Module):
    """
    Encodes features by semantic groups (Pokemon, opponent Pokemon, battle state).
    Includes cross-attention for Pokemon features to learn team synergies.

    Input: (batch, seq, full_feature_dim)
    Output: (batch, seq, aggregated_dim)

    Groups from embedder.group_embedding_sizes:
        - Player Pokemon 0-5: [pokemon_emb_size] * 6
        - Opponent Pokemon 0-5: [opp_pokemon_emb_size] * 6
        - Battle state: [battle_emb_size]
        - Engineered features: [feature_eng_emb_size]
    """
    def __init__(self, group_sizes, hidden_dim=128, aggregated_dim=1024, dropout=0.1):
        super().__init__()
        self.group_sizes = group_sizes
        self.hidden_dim = hidden_dim

        # Per-group encoders
        self.encoders = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(size, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            )
            for size in group_sizes
        ])

        # Initialize per-group encoders
        for encoder in self.encoders:
            linear_layer = encoder[0]  # type: ignore
            torch.nn.init.kaiming_normal_(linear_layer.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(linear_layer.bias, 0)

        # Cross-attention for player Pokemon (first 6 groups)
        # Allows Pokemon to attend to each other (e.g., Incineroar + Rillaboom synergy)
        self.pokemon_cross_attn = torch.nn.MultiheadAttention(
            hidden_dim, num_heads=2, batch_first=True, dropout=dropout
        )
        self.pokemon_norm = torch.nn.LayerNorm(hidden_dim)

        # Aggregate all group embeddings
        self.aggregator = torch.nn.Linear(hidden_dim * len(group_sizes), aggregated_dim)
        torch.nn.init.xavier_normal_(self.aggregator.weight)
        torch.nn.init.constant_(self.aggregator.bias, 0)

    def forward(self, x):
        # x: (batch, seq, full_feature_dim)
        batch, seq, _ = x.shape

        # Encode each group
        group_features = []
        start_idx = 0
        for encoder, size in zip(self.encoders, self.group_sizes):
            group = x[:, :, start_idx:start_idx + size]
            group_features.append(encoder(group))
            start_idx += size

        # Cross-attention among player Pokemon (first 6 groups)
        # This helps the model learn team compositions and synergies
        player_pokemon = torch.stack(group_features[:6], dim=2)  # (batch, seq, 6, hidden)
        player_pokemon_flat = player_pokemon.reshape(batch * seq, 6, -1)  # (batch*seq, 6, hidden)

        attn_out, _ = self.pokemon_cross_attn(
            player_pokemon_flat,
            player_pokemon_flat,
            player_pokemon_flat
        )  # (batch*seq, 6, hidden)

        attn_out = attn_out.reshape(batch, seq, 6, -1)  # (batch, seq, 6, hidden)

        # Apply residual connection and normalization
        for i in range(6):
            group_features[i] = self.pokemon_norm(group_features[i] + attn_out[:, :, i, :])

        # Concatenate all groups and aggregate
        concatenated = torch.cat(group_features, dim=-1)  # (batch, seq, hidden * num_groups)
        return self.aggregator(concatenated)  # (batch, seq, aggregated_dim)


class FlexibleTwoHeadedModel(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        early_layers: list,
        late_layers: list,
        num_attention_heads: int = 4,
        lstm_layers: int = 2,
        num_actions: int = MDBO.action_space(),
        max_seq_len: int = 40,
        dropout: float = 0.1,
        gated_residuals: bool = False,
        early_attention: bool = False,
        late_attention: bool = True,
        use_grouped_encoder: bool = False,
        group_sizes=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.use_grouped_encoder = use_grouped_encoder
        self.hidden_size = early_layers[-1] if early_layers else input_size
        self.num_actions = num_actions
        self.early_attention = early_attention
        self.late_attention = late_attention

        # Select residual block type
        ResBlock = GatedResidualBlock if gated_residuals else ResidualBlock

        # Feature encoding: either grouped or simple linear
        if use_grouped_encoder and group_sizes is not None:
            self.feature_encoder: Optional[GroupedFeatureEncoder] = GroupedFeatureEncoder(
                group_sizes=group_sizes,
                hidden_dim=128,
                aggregated_dim=early_layers[0],
                dropout=dropout
            )
            # early_ff_stack processes [early_layers[0] → early_layers[1] → ...]
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
        else:
            # Simple linear projection
            self.feature_encoder = None
            input_proj = torch.nn.Linear(input_size, early_layers[0])
            torch.nn.init.kaiming_normal_(input_proj.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(input_proj.bias, 0)
            self.input_proj = input_proj

            # early_ff_stack processes [early_layers[0] → early_layers[1] → ...]
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

        # Early attention if enabled
        if early_attention:
            self.early_attn = torch.nn.MultiheadAttention(
                self.hidden_size, num_attention_heads, batch_first=True, dropout=dropout
            )
            self.early_ln = torch.nn.LayerNorm(self.hidden_size)

        # Positional encoding (learned)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        # Bidirectional LSTM (no projection - uses natural bidirectional output)
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        # No lstm_proj - bidirectional output is 2 * hidden_size

        # Skip connection projection (only if grouped encoder is used)
        if use_grouped_encoder:
            self.skip_proj: Optional[torch.nn.Linear] = torch.nn.Linear(self.hidden_size, self.hidden_size * 2)
            torch.nn.init.xavier_normal_(self.skip_proj.weight, gain=0.01)
            torch.nn.init.constant_(self.skip_proj.bias, 0)
        else:
            self.skip_proj = None

        # Late attention if enabled (operates on LSTM output size)
        lstm_output_size = self.hidden_size * 2  # Bidirectional
        if late_attention:
            self.late_attn = torch.nn.MultiheadAttention(
                lstm_output_size, num_attention_heads, batch_first=True, dropout=dropout
            )
            self.late_ln = torch.nn.LayerNorm(lstm_output_size)

        # Build late feedforward stack with residual blocks
        late_ff_layers = []
        prev_size = lstm_output_size
        for h in late_layers:
            late_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.late_ff_stack = (
            torch.nn.Sequential(*late_ff_layers) if late_ff_layers else torch.nn.Identity()
        )

        # Output size after late layers
        output_size = late_layers[-1] if late_layers else lstm_output_size

        # Action head
        self.action_head = torch.nn.Linear(output_size, num_actions)
        torch.nn.init.xavier_normal_(self.action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.action_head.bias, 0)

        # Win prediction head - LayerNorm instead of ReLU to allow negative values
        win_linear1 = torch.nn.Linear(output_size, 128)
        win_linear2 = torch.nn.Linear(128, 1)

        # Initialize with small values to prevent explosion
        torch.nn.init.xavier_normal_(win_linear1.weight, gain=0.01)
        torch.nn.init.constant_(win_linear1.bias, 0)
        torch.nn.init.xavier_normal_(win_linear2.weight, gain=0.01)
        torch.nn.init.constant_(win_linear2.bias, 0)

        self.win_head = torch.nn.Sequential(
            win_linear1,
            torch.nn.LayerNorm(128),  # LayerNorm instead of ReLU
            torch.nn.Dropout(dropout),
            win_linear2,
            torch.nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Feature encoding
        if self.feature_encoder is not None:
            # Grouped encoder path
            ff_out_early = self.feature_encoder(x)  # (batch, seq, early_layers[0])
        else:
            # Simple linear projection path
            ff_out_early = self.input_proj(x)  # (batch, seq, early_layers[0])

        # Early feedforward stack
        ff_out_early = self.early_ff_stack(ff_out_early)  # (batch, seq, hidden_size)

        # Early attention if enabled
        if self.early_attention:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            attn_out, _ = self.early_attn(ff_out_early, ff_out_early, ff_out_early, key_padding_mask=attn_mask)
            ff_out_early = self.early_ln(ff_out_early + attn_out)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        if positions.max() >= self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )

        x_pos = ff_out_early + self.pos_embedding(positions)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        # LSTM (packed)
        lengths = mask.sum(dim=1).long().cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_pos, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        # lstm_out is (batch, seq, hidden_size * 2) - no projection needed

        # Skip connection (only if grouped encoder is used)
        if self.skip_proj is not None:
            ff_out_early_proj = self.skip_proj(ff_out_early)  # (batch, seq, hidden_size) → (batch, seq, hidden_size * 2)
            lstm_out = lstm_out + ff_out_early_proj

        # Late attention if enabled
        if self.late_attention:
            attn_mask = ~mask.bool() if mask is not None else None
            attn_out, _ = self.late_attn(
                lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
            )
            out = self.late_ln(lstm_out + attn_out)
        else:
            out = lstm_out

        # Late feedforward stack
        out = self.late_ff_stack(out)

        # Output heads
        action_logits = self.action_head(out)  # (batch, seq_len, num_actions)
        win_logits = self.win_head(out).squeeze(-1)  # (batch, seq_len), values in [-1, 1]

        return action_logits, win_logits

    def predict(
        self, x: torch.Tensor, mask=None
    ):
        with torch.no_grad():
            action_logits, win_logits = self.forward(x, mask)
            action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs, win_logits


class BehaviorClonePlayer(Player):
    def __init__(
        self,
        teampreview_model_filepath: str,
        action_model_filepath: str,
        win_model_filepath: str,
        teampreview_config: Dict[str, Any],
        action_config: Dict[str, Any],
        win_config: Dict[str, Any],
        battle_format: str = "gen9vgc2023regulationc",
        probabilistic=True,
        **kwargs,
    ):
        # pull in all player information manually
        super().__init__(**kwargs, battle_format=battle_format)
        self._embedder = Embedder(
            format=battle_format, feature_set=Embedder.FULL, omniscient=False
        )
        self._probabilistic = probabilistic
        self._trajectories: Dict[str, list] = {}

        # The models that we use to make predictions
        self.teampreview_model = self._load_model(teampreview_model_filepath, teampreview_config)
        self.action_model = self._load_model(action_model_filepath, action_config)
        self.win_model = self._load_model(win_model_filepath, win_config)

        self._last_message_error: Dict[str, bool] = {}
        self._last_message: Dict[str, str] = {}

        # Track win advantage for dramatic swings
        self._last_win_advantage: Dict[str, float] = {}
        self._win_advantage_threshold = 0.5  # Threshold for "dramatic" swing

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
        self._last_win_advantage = {}

    def _load_model(self, filepath: str, config: Dict[str, Any]) -> FlexibleTwoHeadedModel:
        model = FlexibleTwoHeadedModel(
            input_size=self._embedder.embedding_size,
            early_layers=config["early_layers"],
            late_layers=config["late_layers"],
            num_attention_heads=config["num_attention_heads"],
            lstm_layers=config["lstm_layers"],
            dropout=config["dropout"],
            gated_residuals=config["gated_residuals"],
            early_attention=config["early_attention"],
            late_attention=config["late_attention"],
            use_grouped_encoder=config["use_grouped_encoder"],
            group_sizes=self._embedder.group_embedding_sizes if config["use_grouped_encoder"] else None,
            num_actions=MDBO.action_space(),
            max_seq_len=17,
        ).to(config["device"])
        model.load_state_dict(torch.load(filepath))
        model.eval()
        return model

    def embed_battle_state(self, battle: AbstractBattle) -> List[float]:
        assert isinstance(battle, DoubleBattle)
        assert self._embedder.embedding_size == len(self._embedder.embed(battle))
        return self._embedder.feature_dict_to_vector(self._embedder.embed(battle))

    async def _check_win_advantage_swing(self, battle: DoubleBattle, current_advantage: float) -> None:
        """
        Check if win advantage has dramatically swung and send a message if so.

        Args:
            battle: Current battle state
            current_advantage: Current win advantage prediction (-1 to 1)
        """
        battle_tag = battle.battle_tag

        # Get previous advantage (if exists)
        if battle_tag in self._last_win_advantage:
            prev_advantage = self._last_win_advantage[battle_tag]

            # Check for dramatic positive swing (was losing, now winning)
            if prev_advantage < -self._win_advantage_threshold and current_advantage > self._win_advantage_threshold:
                await self.send_message("skill gap", battle_tag)

            # Check for dramatic negative swing (was winning, now losing)
            elif prev_advantage > self._win_advantage_threshold and current_advantage < -self._win_advantage_threshold:
                await self.send_message("misclick", battle_tag)

        # Update tracked advantage
        self._last_win_advantage[battle_tag] = current_advantage

    def predict(self, traj: torch.Tensor, battle: DoubleBattle) -> Dict[MDBO, float]:
        """
        Given a trajectory tensor and battle, returns a dict of valid actions and their probabilities
        for the last state in the trajectory.
        """
        # Use appropriate model based on battle phase
        model = self.teampreview_model if battle.teampreview else self.action_model

        traj = traj[:, -model.max_seq_len :, :]  # type: ignore
        model.eval()
        with torch.no_grad():
            # Forward pass: get logits for all steps in the trajectory
            action_logits, win_logits = model(
                traj
            )  # shape: (seq_len, num_actions) or (batch, seq_len, num_actions)
            if action_logits.dim() == 3:
                # Remove batch dimension if present
                action_logits = action_logits.squeeze(0)
                win_logits = win_logits.squeeze(0)

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
