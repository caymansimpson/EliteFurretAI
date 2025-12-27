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

from typing import Optional, Tuple, Literal
import torch

from elitefurretai.etl import MDBO


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


class DNN(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512], dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size

        # Build residual blocks
        for size in hidden_sizes:
            layers.append(ResidualBlock(prev_size, size, dropout))
            prev_size = size

        self.backbone = torch.nn.Sequential(*layers)
        self.action_head = torch.nn.Linear(prev_size, MDBO.action_space())

        # Initialize weights
        for layer in self.backbone:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")

        torch.nn.init.xavier_normal_(self.action_head.weight)

    def forward(self, x, masks=None):
        x = self.backbone(x)
        action_logits = self.action_head(x)
        return action_logits


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

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for RL inference with explicit LSTM hidden state management.

        **WHY THIS EXISTS:**
        - BC (Behavioral Cloning) training uses `forward()` which processes entire battle
          trajectories (seq_len=17) at once. The LSTM is bidirectional and state is not
          carried between batches - each trajectory is independent.
        - RL training uses online play where the agent makes one decision at a time
          (seq_len=1). We need to explicitly pass LSTM hidden state between consecutive
          turns of the same battle to maintain temporal coherence.

        **KEY DIFFERENCES FROM forward():**
        1. Accepts hidden_state as input parameter (previous turn's LSTM state)
        2. Returns next_hidden as output (to feed into next turn)
        3. Uses unidirectional LSTM behavior (only processes current timestep)
        4. No packed sequences (seq_len=1 for online inference)
        5. Initializes hidden_state to zeros on first turn if None

        **USAGE IN RL:**
        - Turn 0 (teampreview): Call with hidden_state=None, get initial next_hidden
        - Turn 1+: Call with hidden_state from previous turn, get updated next_hidden
        - Each battle maintains its own hidden_state across turns
        - Reset hidden_state to None when battle ends

        Args:
            x: Input features (batch, seq_len, input_size)
               In RL typically seq_len=1 (current game state only)
            hidden_state: Tuple of (h_n, c_n) from previous turn's LSTM
                         - h_n: (num_layers * 2, batch, lstm_hidden_size) - hidden state
                         - c_n: (num_layers * 2, batch, lstm_hidden_size) - cell state
                         - Note: * 2 because LSTM is bidirectional
                         - Pass None on first turn to initialize with zeros
            mask: Optional padding mask (batch, seq_len) where 1=valid, 0=padding
                  For RL inference with seq_len=1, typically all ones or None

        Returns:
            turn_action_logits: (batch, seq_len, 2025) - Turn action predictions
            teampreview_logits: (batch, seq_len, 90) - Teampreview predictions
            win_logits: (batch, seq_len) - Win prediction in [-1, 1]
            next_hidden: Tuple (h_n, c_n) - Updated LSTM state to pass to next turn
        """
        batch_size, seq_len, _ = x.shape

        # Feature encoding (same as forward())
        if self.feature_encoder is not None:
            ff_out_early = self.feature_encoder(x)
        else:
            ff_out_early = self.input_proj(x)

        # Early feedforward stack
        ff_out_early = self.early_ff_stack(ff_out_early)

        # Teampreview head (branches before LSTM, same as forward())
        teampreview_features = self.teampreview_ff_stack(ff_out_early)

        if self.teampreview_attn is not None:
            attn_mask = ~mask.bool() if mask is not None else None
            tp_attn_out, _ = self.teampreview_attn(
                teampreview_features, teampreview_features, teampreview_features,
                key_padding_mask=attn_mask
            )
            teampreview_features = self.teampreview_ln(teampreview_features + tp_attn_out)  # type: ignore

        teampreview_logits = self.teampreview_head(teampreview_features)

        # Early attention if enabled
        if self.early_attn is not None:
            attn_mask = ~mask.bool() if mask is not None else None
            attn_out, _ = self.early_attn(ff_out_early, ff_out_early, ff_out_early, key_padding_mask=attn_mask)
            ff_out_early = self.early_ln(ff_out_early + attn_out)  # type: ignore

        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        if positions.max() >= self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        x_pos = ff_out_early + self.pos_embedding(positions)

        # LSTM with explicit hidden state management (KEY DIFFERENCE)
        # Initialize hidden state to zeros on first turn (when hidden_state=None)
        if hidden_state is None:
            # Shape: (num_layers * num_directions, batch, hidden_size)
            # For bidirectional LSTM: num_directions=2
            num_layers = self.lstm.num_layers
            num_directions = 2 if self.lstm.bidirectional else 1
            h_0 = torch.zeros(
                num_layers * num_directions,
                batch_size,
                self.lstm.hidden_size,
                device=x.device,
                dtype=x.dtype
            )
            c_0 = torch.zeros(
                num_layers * num_directions,
                batch_size,
                self.lstm.hidden_size,
                device=x.device,
                dtype=x.dtype
            )
            hidden_state = (h_0, c_0)

        # Run LSTM with explicit hidden state
        # lstm_out: (batch, seq_len, lstm_hidden_size * num_directions)
        # next_hidden: tuple of (h_n, c_n) with same shape as hidden_state
        lstm_out, next_hidden = self.lstm(x_pos, hidden_state)

        # Skip connection from early features
        if self.skip_proj is not None:
            ff_out_early_proj = self.skip_proj(ff_out_early)
            lstm_out = lstm_out + ff_out_early_proj
        else:
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

        # Output heads (same as forward())
        turn_action_logits = self.turn_action_head(turn_features)
        win_logits = self.win_head(out).squeeze(-1)

        return turn_action_logits, teampreview_logits, win_logits, next_hidden

    def predict(
        self, x: torch.Tensor, mask=None
    ):
        with torch.no_grad():
            turn_action_logits, teampreview_logits, win_logits = self.forward(x, mask)
            turn_action_probs = torch.softmax(turn_action_logits, dim=-1)
            teampreview_probs = torch.softmax(teampreview_logits, dim=-1)
        return turn_action_probs, teampreview_probs, win_logits
