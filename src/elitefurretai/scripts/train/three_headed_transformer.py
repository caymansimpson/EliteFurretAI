# -*- coding: utf-8 -*-
"""
This script trains a supervised model for teampreview, turn action, and win prediction; if you have collected replays,
you can build a model to try to play like humans.

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
│ Each group: Linear(group_size → X) → LN → ReLU → Dropout        │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ CROSS-GROUP ATTENTION (Pokemon Synergies)                   │ │
│ │                                                             │ │
│ │ Player Pokemon features attend to each other:               │ │
│ │   Pokemon_0 ←→ Pokemon_1 ←→ ... ←→ Pokemon_5                │ │
│ │   (Learns team synergies and interactions)                  │ │
│ │                                                             │ │
│ │ MultiheadAttention(N heads, X dim)                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Aggregate: Concatenate all groups → Linear(14*X = 1792) → Y     │
└─────────----────────────────────────────────────────────────────┘
    |
    v
(batch, seq_len, Y)
    |
    v
┌─────────────────────────────────────────────────────────────────┐
│ FEEDFORWARD STACK #1 (Pre-LSTM, N ResidualBlocks)               │
│   ResidualBlock(Y → Y1)                                         │
│   ResidualBlock(Y1 → Y2)                                        │
|   ...                                                           │
│   ResidualBlock(YN-1 → YN)                                      │
└─────────────────────────────────────────────────────────────────┘
    |
    | ff_out_early: (batch, seq_len, YN)
    |─────────────────────────────────────┬──────────────────────────────┐
    |                                     |                              |
    |                                     |                              v
    |                                     |                 ┌──────────────────────────┐
    |                                     |                 │ TEAMPREVIEW HEAD         │
    |                                     |                 │ (Branches before LSTM)   │
    |                                     |                 │                          │
    |                                     |                 │ Attention (optional)     │
    |                                     |                 │ Residual Block FF layers │
    |                                     |                 │ Linear → 90 actions      │
    |                                     |                 └──────────────────────────┘
    |                                     |                              |
    |                                     |                              v
    |                                     |                      teampreview_logits
    |                                     |                      (batch, seq_len, 90)
    |                                     |
    v                                     |
+ Positional Encoding (YN-dim)            |
    |                                     │
    v                                     │
┌─────────────────────────────────────┐   │
│ BIDIRECTIONAL LSTM (2 layers)       │   │
│   Input: YN                         │   │
│   Hidden: Z per direction           │   │
│   Output: Z (bidirectional)         │   │
│   (No projection needed)            │   │
└─────────────────────────────────────┘   │
    |                                     │
    | lstm_out: (batch, seq_len, Z)       │
    |                                     │
    | Project skip connection:            │
    | skip_proj(ff_out_1): YN → Z         │
    |                                     │  SKIP CONNECTION
    |<────────────────────────────────────┘ (skip_proj(ff_out_1) + lstm_out)
    v
lstm_out = lstm_out + skip_proj(ff_out_1)
    |
    | (batch, seq_len, Z)
    v
┌──────────────────────────────────────------------------------───┐
│ FEEDFORWARD STACK #2 (Post-LSTM, N ResidualBlocks)              │
│   ResidualBlock(Z → Z1)                                         │
│   ResidualBlock(Z1 → Z2)                                        │
|   ...                                                           │
│   ResidualBlock(ZN-1 → ZN)                                      │
└─────────────────────────────────────────────────────────────────┘
    |
    | ff_out_2: (batch, seq_len, ZN)
    v
┌─────────────────────────────────────────┐
│ MULTI-HEAD SELF-ATTENTION (8 heads)     │
│   Query, Key, Value: ff_out_2           │
│   Residual: attn_out + ff_out_2         │
│   LayerNorm                             │
└─────────────────────────────────────────┘
    |
    v
out: (batch, seq_len, ZN)
    |
    ├────────────────────────────────────────┐
    v                                        v
┌────────────------──---────┐      ┌───────────────────┐
│ TURN ACTION HEAD          │      │ WIN HEAD          │
│ (Deep architecture)       │      │ Linear(ZN → 128)  │
│                           │      │ LayerNorm         │
│ ResidualBlock(ZN → T1)    │      │ Dropout           │
│ ResidualBlock(T1 → T2)    │      │ Linear(128 → 1)   │
│ Linear(T2 → 2025 actions) │      │ Tanh              │
└────────────────---------──┘      └───────────────────┘
(batch, seq_len, 2025)             (batch, seq_len)
                                   values ∈ [-1, 1]

Key Features:
• Grouped encoding: Separates Pokemon/Battle features for structured learning
• Cross-attention: Pokemon features attend to each other (team synergies)
• Dual FF stacks: Pre-LSTM (1024→1024→512) and Post-LSTM (1024→512→256)
• Three-headed: Teampreview (90), Turn actions (2025), Win prediction (1)
• Teampreview head: Branches from early features before LSTM (no temporal context needed)
• Hourglass architecture: Wide (1024) → Narrow (512) → Wide (1024) → Narrow (256)
• Larger LSTM: 512-dim hidden states → 1024-dim output (bidirectional)
• Skip connection: FF#1 output (512) projected to 1024, added to LSTM output
• No double ReLU: ResidualBlocks allow negative values for win head
"""

import os.path
import random
import sys
import time
from typing import Any, Dict, Optional, Tuple, Literal
import gc

import torch
import orjson
import wandb

from elitefurretai.model_utils import MDBO, Embedder, OptimizedBattleDataLoader
from elitefurretai.model_utils.train_utils import (
    analyze,
    evaluate,
    format_time,
    topk_cross_entropy_loss,
    focal_topk_cross_entropy_loss,
)


def init_linear_layer(layer: torch.nn.Linear, nonlinearity: Literal['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'selu'] = 'relu') -> None:
    """Initialize a linear layer with Kaiming normal initialization."""
    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity=nonlinearity)
    torch.nn.init.constant_(layer.bias, 0)


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
        init_linear_layer(self.linear)

        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            shortcut_linear = torch.nn.Linear(in_features, out_features)
            init_linear_layer(shortcut_linear)
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
        init_linear_layer(self.linear1)
        init_linear_layer(self.linear2)
        torch.nn.init.xavier_normal_(gate_linear.weight, gain=1.0)
        torch.nn.init.constant_(gate_linear.bias, 0)

        # Projection for residual if dimensions change
        self.shortcut = torch.nn.Sequential()
        if in_features != out_features:
            shortcut_linear = torch.nn.Linear(in_features, out_features)
            init_linear_layer(shortcut_linear)
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
    def __init__(self, group_sizes, hidden_dim=128, aggregated_dim=1024, dropout=0.1, pokemon_attention_heads=2):
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
            init_linear_layer(linear_layer)

        # Cross-attention for player Pokemon (first 6 groups)
        # Allows Pokemon to attend to each other (e.g., Incineroar + Rillaboom synergy)
        self.pokemon_cross_attn = torch.nn.MultiheadAttention(
            hidden_dim, num_heads=pokemon_attention_heads, batch_first=True, dropout=dropout
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


class FlexibleThreeHeadedModel(torch.nn.Module):
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
        turn_head_layers: Optional[list] = None,
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
        self.turn_head_layers = turn_head_layers or []

        # Select residual block type
        ResBlock = GatedResidualBlock if gated_residuals else ResidualBlock

        # Feature encoding: either grouped or simple linear
        if use_grouped_encoder and group_sizes is not None:
            self.feature_encoder: Optional[GroupedFeatureEncoder] = GroupedFeatureEncoder(
                group_sizes=group_sizes,
                hidden_dim=grouped_encoder_hidden_dim,
                aggregated_dim=grouped_encoder_aggregated_dim,
                dropout=dropout,
                pokemon_attention_heads=pokemon_attention_heads
            )
            # early_ff_stack input is aggregated_dim, NOT early_layers[0]
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
            # Simple linear projection
            self.feature_encoder = None
            input_proj = torch.nn.Linear(input_size, early_layers[0])
            init_linear_layer(input_proj)
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

        # Teampreview head (branches from early features before LSTM)
        # Build feedforward stack for teampreview
        teampreview_ff_layers = []
        prev_size = self.hidden_size
        for h in self.teampreview_head_layers:
            teampreview_ff_layers.append(ResBlock(prev_size, h, dropout=teampreview_head_dropout))
            prev_size = h

        teampreview_output_size = self.teampreview_head_layers[-1] if self.teampreview_head_layers else self.hidden_size

        # Optional attention for teampreview head
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

        # Teampreview action head
        self.teampreview_head = torch.nn.Linear(teampreview_output_size, num_teampreview_actions)
        torch.nn.init.xavier_normal_(self.teampreview_head.weight, gain=0.01)
        torch.nn.init.constant_(self.teampreview_head.bias, 0)

        # Early attention if enabled (heads > 0)
        if early_attention_heads > 0:
            self.early_attn: Optional[torch.nn.MultiheadAttention] = torch.nn.MultiheadAttention(
                self.hidden_size, early_attention_heads, batch_first=True, dropout=dropout
            )
            self.early_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(self.hidden_size)
        else:
            self.early_attn = None
            self.early_ln = None

        # Positional encoding (learned)
        self.pos_embedding = torch.nn.Embedding(max_seq_len, self.hidden_size)

        # Bidirectional LSTM (no projection - uses natural bidirectional output)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        # LSTM output is 2 * lstm_hidden_size due to bidirectionality

        # Skip connection projection (if dimensions don't match)
        lstm_output_size = lstm_hidden_size * 2
        if self.hidden_size != lstm_output_size:
            self.skip_proj: Optional[torch.nn.Linear] = torch.nn.Linear(self.hidden_size, lstm_output_size)
            torch.nn.init.xavier_normal_(self.skip_proj.weight, gain=0.01)
            torch.nn.init.constant_(self.skip_proj.bias, 0)
        else:
            self.skip_proj = None

        # Late attention if enabled (operates on LSTM output size)
        if late_attention_heads > 0:
            self.late_attn: Optional[torch.nn.MultiheadAttention] = torch.nn.MultiheadAttention(
                lstm_output_size, late_attention_heads, batch_first=True, dropout=dropout
            )
            self.late_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(lstm_output_size)
        else:
            self.late_attn = None
            self.late_ln = None

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

        # Turn action head (for in-battle decisions)
        # Build feedforward stack for turn head
        turn_ff_layers = []
        prev_size = output_size
        for h in self.turn_head_layers:
            turn_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h

        turn_output_size = self.turn_head_layers[-1] if self.turn_head_layers else output_size

        self.turn_ff_stack = (
            torch.nn.Sequential(*turn_ff_layers)
            if turn_ff_layers
            else torch.nn.Identity()
        )

        # Final linear layer for turn actions
        self.turn_action_head = torch.nn.Linear(turn_output_size, num_actions)
        torch.nn.init.xavier_normal_(self.turn_action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.turn_action_head.bias, 0)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through three-headed model.

        Returns:
            turn_action_logits: (batch, seq_len, 2025) - Turn action predictions
            teampreview_logits: (batch, seq_len, 90) - Teampreview action predictions
            win_logits: (batch, seq_len) - Win prediction
        """
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

        # Process teampreview head (branches before LSTM)
        teampreview_features = self.teampreview_ff_stack(ff_out_early)

        # Optional attention for teampreview
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

        teampreview_logits = self.teampreview_head(teampreview_features)  # (batch, seq, 90)

        # Early attention if enabled
        if self.early_attn is not None:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            attn_out, _ = self.early_attn(ff_out_early, ff_out_early, ff_out_early, key_padding_mask=attn_mask)
            ff_out_early = self.early_ln(ff_out_early + attn_out)  # type: ignore

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
        # lstm_out is (batch, seq, lstm_hidden_size * 2) - bidirectional output

        # Skip connection from early features
        if self.skip_proj is not None:
            # Dimensions don't match, use projection
            ff_out_early_proj = self.skip_proj(ff_out_early)
            lstm_out = lstm_out + ff_out_early_proj
        else:
            # Dimensions match, direct addition
            lstm_out = lstm_out + ff_out_early

        # Late attention if enabled
        if self.late_attn is not None:
            attn_mask = ~mask.bool() if mask is not None else None
            attn_out, _ = self.late_attn(
                lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask
            )
            out = self.late_ln(lstm_out + attn_out)  # type: ignore
        else:
            out = lstm_out

        # Late feedforward stack
        out = self.late_ff_stack(out)

        # Process turn head with feedforward stack
        turn_features = self.turn_ff_stack(out)

        # Output heads
        turn_action_logits = self.turn_action_head(turn_features)  # (batch, seq_len, 2025)
        win_logits = self.win_head(out).squeeze(-1)  # (batch, seq_len), values in [-1, 1]

        return turn_action_logits, teampreview_logits, win_logits

    def predict(
        self, x: torch.Tensor, mask=None
    ):
        with torch.no_grad():
            turn_action_logits, teampreview_logits, win_logits = self.forward(x, mask)
            turn_action_probs = torch.softmax(turn_action_logits, dim=-1)
            teampreview_probs = torch.softmax(teampreview_logits, dim=-1)
        return turn_action_probs, teampreview_probs, win_logits


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    prev_steps: int,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    scaler=None,
) -> Dict[str, Any]:
    model.train()
    running_loss = 0.0
    running_turn_loss = 0.0
    running_teampreview_loss = 0.0
    running_win_loss = 0.0
    running_entropy = 0.0
    steps = 0
    num_batches = 0
    start = time.time()

    # Gradient accumulation setup to reduce memory pressure
    accumulation_steps = config.get("accumulation_steps", 1)
    accumulation_counter = 0

    for batch in dataloader:

        # Transfer data to the right device
        if config['device'] == 'cuda':
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        else:
            batch = {k: v.to(config['device']) for k, v in batch.items()}

        states = batch["states"].to(torch.float32)
        actions = batch["actions"]
        action_masks = batch["action_masks"]
        wins = batch["wins"].to(torch.float32)
        masks = batch["masks"]

        autocast = torch.amp.autocast if config['device'] == 'cuda' else torch.autocast  # type: ignore
        with autocast(config['device']):

            # Forward pass - returns three outputs
            turn_action_logits, teampreview_logits, win_logits = model(states, masks)

            # Determine which samples are teampreview vs turn decisions
            teampreview_mask = states[:, :, config["teampreview_idx"]] == 1  # (batch, seq)
            turn_mask = ~teampreview_mask  # (batch, seq)

            # Mask action logits appropriately
            # Teampreview: mask first 90 actions, Turn: mask all 2025 actions
            masked_turn_logits = turn_action_logits.masked_fill(
                ~action_masks.bool(), float("-inf")
            )
            # For teampreview, only use first 90 dimensions
            teampreview_action_masks = action_masks[:, :, :MDBO.teampreview_space()]
            masked_teampreview_logits = teampreview_logits.masked_fill(
                ~teampreview_action_masks.bool(), float("-inf")
            )

            # Flatten and filter valid samples
            valid_mask = masks.bool()  # (batch, seq)

            # Detect force switch states - any mon has force_switch=1
            # and mask them to force learning actions
            force_switch_mask = torch.zeros_like(turn_mask)
            for fs_idx in config["force_switch_indices"]:
                force_switch_mask = force_switch_mask | (states[:, :, fs_idx] > 0.5)

            # Conditionally exclude force switches from turn training based on config
            if not config.get("keep_force_switch", True):
                turn_valid_mask = valid_mask & turn_mask & ~force_switch_mask
            else:
                turn_valid_mask = valid_mask & turn_mask

            if turn_valid_mask.any():
                flat_turn_logits = masked_turn_logits[turn_valid_mask]
                flat_turn_actions = actions[turn_valid_mask]
                flat_turn_wins = wins[turn_valid_mask]
                flat_turn_win_logits = win_logits[turn_valid_mask]

                # Choose loss function based on config
                loss_type = config.get("loss_type", "top3")
                if loss_type == "focal":
                    turn_loss = focal_topk_cross_entropy_loss(
                        flat_turn_logits,
                        flat_turn_actions,
                        weights=None,
                        k=config.get("train_topk_k", 2025),
                        gamma=config.get("focal_gamma", 2.0),
                        alpha=config.get("focal_alpha", 0.25),
                    )
                else:  # "top3" or default
                    turn_loss = topk_cross_entropy_loss(
                        flat_turn_logits,
                        flat_turn_actions,
                        weights=None,
                        k=config.get("train_topk_k", 3),
                    )

                turn_win_loss = torch.nn.functional.mse_loss(flat_turn_win_logits, flat_turn_wins.float())

                # Compute entropy for regularization (encourages exploration)
                turn_probs = torch.nn.functional.softmax(flat_turn_logits, dim=-1)
                turn_entropy = -(turn_probs * torch.log(turn_probs + 1e-10)).sum(dim=-1).mean()
            else:
                turn_loss = torch.tensor(0.0, device=states.device)
                turn_win_loss = torch.tensor(0.0, device=states.device)
                turn_entropy = torch.tensor(0.0, device=states.device)

            # Process teampreview samples
            teampreview_valid_mask = valid_mask & teampreview_mask
            if teampreview_valid_mask.any():
                flat_tp_logits = masked_teampreview_logits[teampreview_valid_mask]
                flat_tp_actions = actions[teampreview_valid_mask]
                flat_tp_wins = wins[teampreview_valid_mask]
                flat_tp_win_logits = win_logits[teampreview_valid_mask]

                # Note: teampreview actions should already be in [0, 90) range
                teampreview_loss = torch.nn.functional.cross_entropy(flat_tp_logits, flat_tp_actions)
                teampreview_win_loss = torch.nn.functional.mse_loss(flat_tp_win_logits, flat_tp_wins.float())
            else:
                teampreview_loss = torch.tensor(0.0, device=states.device)
                teampreview_win_loss = torch.tensor(0.0, device=states.device)

            # Combined loss with configurable weights
            # Entropy regularization: negative entropy encourages diversity (higher entropy = more uniform distribution)
            entropy_loss = -turn_entropy if config.get("entropy_weight", 0.0) > 0 else torch.tensor(0.0, device=states.device)

            loss = (
                config["turn_loss_weight"] * turn_loss
                + config["teampreview_loss_weight"] * teampreview_loss
                + config["win_loss_weight"] * (turn_win_loss + teampreview_win_loss)
                + config.get("entropy_weight", 0.0) * entropy_loss
            )

            # Track number of samples
            num_turn_samples = turn_valid_mask.sum().item()
            num_tp_samples = teampreview_valid_mask.sum().item()

            # Scale loss by accumulation steps to maintain gradient magnitude
            loss = loss / accumulation_steps

        # Backpropagation with mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulation_counter += 1

        # Only update weights every accumulation_steps
        if accumulation_counter >= accumulation_steps:
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()

            optimizer.zero_grad()
            accumulation_counter = 0

        # Metrics (multiply back to get actual loss)
        running_loss += loss.item() * accumulation_steps
        running_turn_loss += turn_loss.item()
        running_teampreview_loss += teampreview_loss.item()
        running_win_loss += (turn_win_loss.item() + teampreview_win_loss.item())
        running_entropy += turn_entropy.item() if turn_valid_mask.any() else 0.0
        steps += (num_turn_samples + num_tp_samples)
        num_batches += 1

        # Logging progress (only on actual optimizer steps)
        if num_batches % (100 * accumulation_steps) == 0:
            wandb.log({"Total Steps": prev_steps + steps, "grad_norm": grad_norm.item()})  # type: ignore

            # Periodic Python garbage collection every 100 batches
            gc.collect()

        if num_batches % (10 * accumulation_steps) == 0:
            wandb.log(
                {
                    "Total Steps": prev_steps + steps,
                    "train_loss": running_loss / num_batches,
                    "train_turn_loss": running_turn_loss / num_batches,
                    "train_teampreview_loss": running_teampreview_loss / num_batches,
                    "train_win_loss": running_win_loss / num_batches,
                    "train_entropy": running_entropy / num_batches,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        # Print progress
        assert dataloader.batch_size is not None
        batch_size = dataloader.batch_size
        time_taken = format_time(time.time() - start)
        time_per_batch = (time.time() - start) * 1.0 / (num_batches + 1)
        time_left = format_time((len(dataloader) - num_batches) * time_per_batch)
        processed = f"Processed {num_batches * batch_size} battles/trajectories ({round(num_batches * 100.0 / len(dataloader), 2)}%) in {time_taken}"
        left = f" with an estimated {time_left} left in this epoch"
        print("\033[2K\r" + processed + left, end="")

    time_taken = format_time(time.time() - start)
    print("\033[2K\rDone training in " + time_taken)

    return {
        "loss": running_loss / num_batches,
        "steps": steps,
        "turn_loss": running_turn_loss / num_batches,
        "teampreview_loss": running_teampreview_loss / num_batches,
        "win_loss": running_win_loss / num_batches,
        "entropy": running_entropy / num_batches,
    }


def initialize(config):
    # Wandb defaults
    wandb.init(
        project="elitefurretai-hydreigon",
        config=config,
        settings=wandb.Settings(
            x_service_wait=30,  # Increase service wait time
            start_method="thread"  # Use thread instead of fork
        )
    )
    try:
        # Try normal symlink first (fast)
        wandb.save(__file__)
    except OSError as e:
        if "WinError 1314" in str(e) or "privilege" in str(e).lower():
            try:
                # Fallback to copy method
                wandb.save(__file__, policy="now")
                print("Note: Using file copy instead of symlink for wandb")
            except Exception as copy_error:
                print(f"Warning: Could not save script to wandb: {copy_error}")
        else:
            raise  # Re-raise if it's a different error

    # Set Seeds
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(config["seed"]))
    random.seed(int(config["seed"]))


def main(train_path, test_path, val_path, config={}):

    print("Starting!")
    default_config: Dict[str, Any] = {
        # Training config to optimize speed
        "worker_batch_size": 64,
        "num_workers": 7,
        "prefetch_factor": 8,
        "files_per_worker": 3,
        "persistent_workers": True,

        # Basic Training Params
        "learning_rate": 5e-5,
        "optimizer": "AdamW",
        "num_epochs": 30,

        # Regularization
        "batch_size": 512,
        "dropout": 0.1,
        "weight_decay": 1e-5,
        "max_grad_norm": 2.0,
        "teampreview_head_dropout": 0.1,

        # Loss Weights (three separate heads)
        "teampreview_loss_weight": 1,
        "turn_loss_weight": 1,
        "win_loss_weight": 1,
        "keep_force_switch": True,  # If True, include force switch examples from training
        "entropy_weight": 0.0,  # Entropy regularization to encourage prediction diversity (0.0 = off, try 0.01-0.1)

        # Loss Function Type
        "loss_type": "top3",  # "top3" (standard topk CE) or "focal" (focal loss for hard examples)
        "train_topk_k": 3,  # Number of top predictions to consider (2025 = all actions, 3 = top-3 only)
        "focal_gamma": 2.0,  # Focal loss focusing parameter (higher = more focus on hard examples)
        "focal_alpha": 0.25,  # Focal loss weighting parameter

        # Architecture - Backbone
        "gated_residuals": False,
        "use_grouped_encoder": True,
        "grouped_encoder_hidden_dim": 512,
        "grouped_encoder_aggregated_dim": 4096,
        "pokemon_attention_heads": 8,
        "early_layers": [4096, 4096, 2048, 2048, 1024],
        "early_attention_heads": 8,

        # Win/Action Head Architecture
        "lstm_layers": 3,
        "lstm_hidden_size": 1024,
        "late_layers": [2048, 1024, 1024, 512],
        "late_attention_heads": 16,

        # Architecture - Teampreview Head
        "teampreview_head_layers": [512, 256],
        "teampreview_attention_heads": 8,

        # Architecture - Turn Action Head
        "turn_head_layers": [512, 512, 512],

        # Other
        "device": (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
        "save_path": "data/models/",
        "seed": 21,
    }

    # Update config with any overrides provided in cfg
    for k, v in default_config.items():
        if k not in config:
            config[k] = v

    config["accumulation_steps"] = int(config["batch_size"] // config["worker_batch_size"])
    print(f"Starting training with config: {config}")
    initialize(config)

    # Initialize Embedder and find indices of special features; these will be used
    # for weighting training and analyzing model performance
    embedder = Embedder(
        format="gen9vgc2023regulationc",
        feature_set=Embedder.FULL,
        omniscient=False
    )
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    config["teampreview_idx"] = feature_names["teampreview"]
    config["force_switch_indices"] = [
        feature_names[f"MON:{j}:force_switch"] for j in range(6)
    ]
    print(f"Embedder initialized. Embedding[{embedder.embedding_size}] on {config['device']}")

    print("Loading datasets...")
    start = time.time()

    train_loader = OptimizedBattleDataLoader(
        train_path,
        embedder=embedder,
        num_workers=config["num_workers"],
        prefetch_factor=config["prefetch_factor"],
        files_per_worker=config["files_per_worker"],
        persistent_workers=config["persistent_workers"],
    )
    test_loader = OptimizedBattleDataLoader(test_path, embedder=embedder, batch_size=config["worker_batch_size"], num_workers=4, prefetch_factor=2, files_per_worker=1)
    val_loader = OptimizedBattleDataLoader(val_path, embedder=embedder, batch_size=config["worker_batch_size"], num_workers=4, prefetch_factor=2, files_per_worker=1)

    # Initialize model with flexible architecture
    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config["lstm_layers"],
        lstm_hidden_size=config["lstm_hidden_size"],
        dropout=config["dropout"],
        gated_residuals=config["gated_residuals"],
        early_attention_heads=config["early_attention_heads"],
        late_attention_heads=config["late_attention_heads"],
        use_grouped_encoder=config["use_grouped_encoder"],
        group_sizes=embedder.group_embedding_sizes if config["use_grouped_encoder"] else None,
        grouped_encoder_hidden_dim=config["grouped_encoder_hidden_dim"],
        grouped_encoder_aggregated_dim=config["grouped_encoder_aggregated_dim"],
        pokemon_attention_heads=config["pokemon_attention_heads"],
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        teampreview_head_layers=config["teampreview_head_layers"],
        teampreview_head_dropout=config["teampreview_head_dropout"],
        teampreview_attention_heads=config["teampreview_attention_heads"],
        turn_head_layers=config["turn_head_layers"],
        max_seq_len=17,
    ).to(config["device"])
    wandb.watch(model, log="all", log_freq=1000)

    # Count Parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Finished loading data and model! Total trainable parameters: {num_params:,}")
    wandb.log({"model_parameters": num_params})
    print("Starting training...")

    # Initialize optimizer
    optimizer_class = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }[config["optimizer"]]

    optimizer = optimizer_class(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if config['device'] == 'cuda' else None  # type: ignore

    print("Initialized model! Starting training...")

    # Training loop
    start, steps = time.time(), 0
    for epoch in range(config["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, steps, optimizer, config, scaler)

        # Evaluate with native three-headed support
        metrics = evaluate(
            model,
            test_loader,
            config["device"],
            has_teampreview_head=True,
            teampreview_idx=config["teampreview_idx"],
            config=config,
        )
        steps += train_metrics["steps"]
        test_loss = (
            metrics["win_mse"] * config["win_loss_weight"]
            + metrics.get("teampreview_top3_loss", 0) * config["teampreview_loss_weight"]
            + metrics.get("turn_top3_loss", 0) * config["turn_loss_weight"]
        )

        log = {
            "Total Steps": steps,
            "Train Loss": train_metrics["loss"],
            "Train Win Loss": train_metrics["win_loss"],
            "Train Turn Loss": train_metrics["turn_loss"],
            "Train Teampreview Loss": train_metrics["teampreview_loss"],
            "Test Loss": test_loss,
            "Test Win Corr": metrics["win_corr"],
            "Test Win MSE": metrics["win_mse"],
            "Test Teampreview Top3 Loss": metrics.get("teampreview_top3_loss", 0),
            "Test Teampreview Top1": metrics.get("teampreview_top1_acc", 0),
            "Test Teampreview Top3": metrics.get("teampreview_top3_acc", 0),
            "Test Teampreview Top5": metrics.get("teampreview_top5_acc", 0),
            "Test Turn Top3 Loss": metrics.get("turn_top3_loss", 0),
            "Test Turn Top1": metrics.get("turn_top1_acc", 0),
            "Test Turn Top3": metrics.get("turn_top3_acc", 0),
            "Test Turn Top5": metrics.get("turn_top5_acc", 0),
            # Action-type specific metrics
            "Test MOVE Top1": metrics.get("move_top1_acc", 0),
            "Test MOVE Top3": metrics.get("move_top3_acc", 0),
            "Test MOVE Top5": metrics.get("move_top5_acc", 0),
            "Test SWITCH Top1": metrics.get("switch_top1_acc", 0),
            "Test SWITCH Top3": metrics.get("switch_top3_acc", 0),
            "Test SWITCH Top5": metrics.get("switch_top5_acc", 0),
            "Test BOTH Top1": metrics.get("both_top1_acc", 0),
            "Test BOTH Top3": metrics.get("both_top3_acc", 0),
            "Test BOTH Top5": metrics.get("both_top5_acc", 0),
        }

        print(f"Epoch #{epoch + 1}:")
        for metric, value in log.items():
            print(f"=> {metric:<30}: {value:>10.3f}")

        wandb.log(log)

        total_time = time.time() - start
        time_taken = format_time(total_time)
        time_left = format_time(
            (config["num_epochs"] - epoch - 1) * total_time / (epoch + 1)
        )
        print(f"=> Time thus far: {time_taken} // ETA: {time_left}")
        print()

        scheduler.step(
            float(
                metrics["win_mse"] * config["win_loss_weight"]
                + metrics.get("teampreview_top3_loss", 0) * config["teampreview_loss_weight"]
                + metrics.get("turn_top3_loss", 0) * config["turn_loss_weight"]
            )
        )

    # Save model with config embedded
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    save_path = os.path.join(config["save_path"], f"{wandb.run.name}.pt")  # type: ignore
    torch.save(save_dict, save_path)
    print(f"\nModel and config saved to {save_path}")

    print("\nEvaluating on Validation Dataset:")
    metrics = evaluate(
        model,
        val_loader,
        config["device"],
        has_teampreview_head=True,
        teampreview_idx=config["teampreview_idx"],
        config=config,
    )
    val_log = {
        "Total Steps": steps,
        "Validation Loss": (
            metrics["win_mse"] * config["win_loss_weight"]
            + metrics.get("teampreview_top3_loss", 0) * config["teampreview_loss_weight"]
            + metrics.get("turn_top3_loss", 0) * config["turn_loss_weight"]
        ),
        "Validation Win Corr": metrics["win_corr"],
        "Validation Win MSE": metrics["win_mse"],
        "Validation Teampreview Top3 Loss": metrics.get("teampreview_top3_loss", 0),
        "Validation Teampreview Top1": metrics.get("teampreview_top1_acc", 0),
        "Validation Teampreview Top3": metrics.get("teampreview_top3_acc", 0),
        "Validation Teampreview Top5": metrics.get("teampreview_top5_acc", 0),
        "Validation Turn Top3 Loss": metrics.get("turn_top3_loss", 0),
        "Validation Turn Top1": metrics.get("turn_top1_acc", 0),
        "Validation Turn Top3": metrics.get("turn_top3_acc", 0),
        "Validation Turn Top5": metrics.get("turn_top5_acc", 0),
    }

    for metric, value in val_log.items():
        print(f"==> {metric:<30}: {value:>10.3f}")

    wandb.log(val_log)

    print("\nAnalyzing...")
    analyze(
        model,
        val_loader,
        device=config["device"],
        has_teampreview_head=True,
        teampreview_idx=config["teampreview_idx"],
        force_switch_indices=config["force_switch_indices"],
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python three_headed_transformer.py <data_directory>")
        sys.exit(1)
    elif len(sys.argv) == 2:
        main(
            os.path.join(sys.argv[1], "train"),
            os.path.join(sys.argv[1], "test"),
            os.path.join(sys.argv[1], "val"),
        )
    elif len(sys.argv) == 3:
        with open(sys.argv[2], 'rb') as f:
            cfg = orjson.loads(f.read())
        main(
            os.path.join(sys.argv[1], "train"),
            os.path.join(sys.argv[1], "test"),
            os.path.join(sys.argv[1], "val"),
            cfg
        )
