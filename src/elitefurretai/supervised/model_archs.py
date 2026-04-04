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

import math
from typing import List, Literal, Optional, Tuple

import torch

from elitefurretai.etl import MDBO
from elitefurretai.etl.embedder import Embedder


def init_linear_layer(
    layer: torch.nn.Linear,
    nonlinearity: Literal[
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "selu",
    ] = "relu",
) -> None:
    """Initialize a linear layer with Kaiming normal initialization."""
    torch.nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity=nonlinearity)
    torch.nn.init.constant_(layer.bias, 0)


def twohot_encode(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Encode scalar values as two-hot targets over a support vector.

    Used by the distributional value head (C51). Each scalar value is represented
    as a probability distribution over two adjacent bins, with linear interpolation.

    Args:
        values: Scalar values to encode, any shape
        support: 1D tensor of bin centers (e.g. linspace(-1, 1, 51))

    Returns:
        Two-hot encoded targets with shape (*values.shape, len(support))
    """
    values = values.clamp(support[0], support[-1])
    bin_width = support[1] - support[0]
    lower_idx = ((values - support[0]) / bin_width).floor().long()
    lower_idx = lower_idx.clamp(0, len(support) - 2)
    upper_idx = lower_idx + 1
    upper_weight = (values - support[lower_idx]) / bin_width
    lower_weight = 1.0 - upper_weight

    targets = torch.zeros(*values.shape, len(support), device=values.device)
    targets.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    targets.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return targets


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

    def __init__(
        self,
        group_sizes,
        feature_names: List[str],
        num_abilities: int,
        num_items: int,
        num_species: int,
        num_moves: int,
        hidden_dim=128,
        aggregated_dim=1024,
        dropout=0.1,
        pokemon_attention_heads=2,
        ability_embed_dim: int = 16,
        item_embed_dim: int = 16,
        species_embed_dim: int = 32,
        move_embed_dim: int = 16,
        number_bank_hp_bins: int = 100,
        number_bank_stat_bins: int = 600,
        number_bank_power_bins: int = 250,
        number_bank_embedding_dim: int = 16,
        number_bank_damage_bins: int = 600,
        number_bank_damage_embed_dim: int = 4,
        number_bank_turn_bins: int = 40,
        number_bank_turn_embed_dim: int = 16,
        number_bank_rating_bins: int = 100,
        number_bank_rating_embed_dim: int = 16,
    ):
        super().__init__()
        self.group_sizes = group_sizes
        self.hidden_dim = hidden_dim
        self.entity_id_encoder = EntityIDEncoder(
            feature_names=feature_names,
            group_sizes=group_sizes,
            num_abilities=num_abilities,
            num_items=num_items,
            num_species=num_species,
            num_moves=num_moves,
            ability_embed_dim=ability_embed_dim,
            item_embed_dim=item_embed_dim,
            species_embed_dim=species_embed_dim,
            move_embed_dim=move_embed_dim,
        )
        self.number_bank = NumberBankEncoder(
            feature_names=feature_names,
            group_sizes=group_sizes,
            hp_bins=number_bank_hp_bins,
            stat_bins=number_bank_stat_bins,
            power_bins=number_bank_power_bins,
            embed_dim=number_bank_embedding_dim,
            damage_bins=number_bank_damage_bins,
            damage_embed_dim=number_bank_damage_embed_dim,
            turn_bins=number_bank_turn_bins,
            turn_embed_dim=number_bank_turn_embed_dim,
            rating_bins=number_bank_rating_bins,
            rating_embed_dim=number_bank_rating_embed_dim,
        )

        # Compute effective per-group input sizes after all embedding expansions.
        # Entity ID encoder and number bank operate on the original tensor positions
        # independently, but their expansions compose. We compute the final size by
        # tracking how each scalar replacement changes the group size.
        effective_sizes = list(self.entity_id_encoder.group_output_sizes)
        # Number bank was built on original group_sizes. Its output sizes account
        # for its own expansions. The final size is additive because entity IDs
        # (ability_id, item_id) don't overlap with number-bank patterns.
        for i in range(len(effective_sizes)):
            nb_expansion = self.number_bank.group_output_sizes[i] - group_sizes[i]
            effective_sizes[i] += nb_expansion

        # Per-group encoders
        self.encoders = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(size, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                )
                for size in effective_sizes
            ]
        )

        # Initialize per-group encoders
        for encoder in self.encoders:
            linear_layer = encoder[0]  # type: ignore
            init_linear_layer(linear_layer)

        # Cross-attention for player Pokemon (first 6 groups)
        # Allows Pokemon to attend to each other (e.g., Incineroar + Rillaboom synergy)
        self.pokemon_cross_attn = torch.nn.MultiheadAttention(
            hidden_dim,
            num_heads=pokemon_attention_heads,
            batch_first=True,
            dropout=dropout,
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
        for group_idx, (encoder, size) in enumerate(
            zip(self.encoders, self.group_sizes)
        ):
            group = x[:, :, start_idx : start_idx + size]
            # Apply entity-ID and number-bank embedding expansions in one pass.
            group = self._dual_expand(group, group_idx)
            group_features.append(encoder(group))
            start_idx += size

        # Cross-attention among player Pokemon (first 6 groups)
        # This helps the model learn team compositions and synergies
        player_pokemon = torch.stack(group_features[:6], dim=2)  # (batch, seq, 6, hidden)
        player_pokemon_flat = player_pokemon.reshape(
            batch * seq, 6, -1
        )  # (batch*seq, 6, hidden)

        attn_out, _ = self.pokemon_cross_attn(
            player_pokemon_flat, player_pokemon_flat, player_pokemon_flat
        )  # (batch*seq, 6, hidden)

        attn_out = attn_out.reshape(batch, seq, 6, -1)  # (batch, seq, 6, hidden)

        # Apply residual connection and normalization
        for i in range(6):
            group_features[i] = self.pokemon_norm(group_features[i] + attn_out[:, :, i, :])

        # Concatenate all groups and aggregate
        concatenated = torch.cat(
            group_features, dim=-1
        )  # (batch, seq, hidden * num_groups)
        return self.aggregator(concatenated)  # (batch, seq, aggregated_dim)

    def _dual_expand(self, x: torch.Tensor, group_idx: int) -> torch.Tensor:
        """Expand entity IDs and number bank features in a single pass.

        Both encoders reference local indices from the *original* group tensor.
        We iterate through all positions left-to-right, applying the appropriate
        expansion for each position, ensuring no index drift.
        """
        eid_map = {
            idx: etype
            for idx, etype in self.entity_id_encoder._group_maps[group_idx]
        }
        nb_map = {
            entry[0]: entry
            for entry in self.number_bank._group_maps[group_idx]
        }

        group_size = x.shape[2]
        parts: List[torch.Tensor] = []
        prev_end = 0

        # Merge all replacement positions
        all_positions = sorted(set(eid_map.keys()) | set(nb_map.keys()))

        for pos in all_positions:
            # Append passthrough features before this position
            if pos > prev_end:
                parts.append(x[:, :, prev_end:pos])

            if pos in eid_map:
                # Entity ID replacement
                raw_id = x[:, :, pos].long().clamp(min=0)
                emb_layer = self.entity_id_encoder._get_embedding(eid_map[pos])
                parts.append(emb_layer(raw_id))
            elif pos in nb_map:
                # Number bank replacement
                local_idx, bank_name, min_val, max_val, n_bins = nb_map[pos]
                raw = x[:, :, local_idx]
                clamped = raw.clamp(min=min_val, max=max_val)
                bucket = ((clamped - min_val) / (max_val - min_val) * n_bins).long()
                bucket = bucket.clamp(0, n_bins)
                bank = self.number_bank._get_bank(bank_name)
                parts.append(bank(bucket))

            prev_end = pos + 1

        # Append remaining features
        if prev_end < group_size:
            parts.append(x[:, :, prev_end:])

        return torch.cat(parts, dim=-1)


class NumberBankEncoder(torch.nn.Module):
    """Replaces raw float inputs for selected features with learned embedding lookups.

    Numerical features (HP%, stats, base power) are discretized into buckets and
    each bucket gets a learned embedding vector.  Non-numerical features pass through
    unchanged.  This is applied *inside* each per-group encoder of
    ``GroupedFeatureEncoder`` so the Embedder output format does not change.

    The caller must supply ``feature_names`` (sorted list from ``Embedder.feature_names``)
    and ``group_sizes`` so that this module can identify which raw-float positions
    correspond to HP/stats/power features by matching name patterns.
    """

    # Name patterns used to classify numerical features.
    HP_PATTERNS = ("current_hp_fraction", "HP_FRAC", "PERC_HP_LEFT", "OPP_PERC_HP_LEFT")
    STAT_PATTERNS = ("STAT:", "STAT_MIN:", "STAT_MAX:")
    POWER_PATTERNS = ("base_power", "BASE_POWER")
    DAMAGE_PATTERNS = ("EST_DAMAGE_MIN:", "EST_DAMAGE_MAX:")
    TURN_PATTERNS = ("turn",)  # exact match handled in _classify_feature
    RATING_PATTERNS = ("p1rating", "p2rating")

    def __init__(
        self,
        feature_names: List[str],
        group_sizes: List[int],
        hp_bins: int = 100,
        stat_bins: int = 600,
        power_bins: int = 250,
        embed_dim: int = 16,
        damage_bins: int = 600,
        damage_embed_dim: int = 4,
        turn_bins: int = 40,
        turn_embed_dim: int = 16,
        rating_bins: int = 100,
        rating_embed_dim: int = 16,
    ):
        super().__init__()
        self.hp_bins = hp_bins
        self.stat_bins = stat_bins
        self.power_bins = power_bins
        self.damage_bins = damage_bins
        self.turn_bins = turn_bins
        self.rating_bins = rating_bins
        self.embed_dim = embed_dim
        self.damage_embed_dim = damage_embed_dim
        self.turn_embed_dim = turn_embed_dim
        self.rating_embed_dim = rating_embed_dim

        # Embedding banks (+1 for clamp-to-edge)
        self.hp_bank = torch.nn.Embedding(hp_bins + 1, embed_dim)
        self.stat_bank = torch.nn.Embedding(stat_bins + 1, embed_dim)
        self.power_bank = torch.nn.Embedding(power_bins + 1, embed_dim)
        self.damage_bank = torch.nn.Embedding(damage_bins + 1, damage_embed_dim)
        self.turn_bank = torch.nn.Embedding(turn_bins + 1, turn_embed_dim)
        self.rating_bank = torch.nn.Embedding(rating_bins + 1, rating_embed_dim)

        # Build per-group maps: for each group, which feature positions are
        # numerical and which bank+range they map to.
        # Each entry: (local_index, bank_name, min_val, max_val, n_bins)
        self._group_maps: List[List[Tuple[int, str, float, float, int]]] = []

        # Also compute new per-group input sizes (after replacing scalars with embeddings)
        self._group_output_sizes: List[int] = []

        offset = 0
        for gsize in group_sizes:
            gmap: List[Tuple[int, str, float, float, int]] = []
            for local_idx in range(gsize):
                global_idx = offset + local_idx
                if global_idx >= len(feature_names):
                    break
                name = feature_names[global_idx]
                entry = self._classify_feature(name)
                if entry is not None:
                    gmap.append((local_idx, *entry))
            self._group_maps.append(gmap)
            # New size = original - n_replaced + sum of per-feature embed_dims
            embed_expansion = sum(
                self._embed_dim_for_entry(entry) for entry in gmap
            )
            n_replaced = len(gmap)
            self._group_output_sizes.append(gsize - n_replaced + embed_expansion)
            offset += gsize

    @property
    def group_output_sizes(self) -> List[int]:
        """Per-group feature dimension after number-bank expansion."""
        return self._group_output_sizes

    def _classify_feature(self, name: str) -> Optional[Tuple[str, float, float, int]]:
        """Return (bank_name, min_val, max_val, n_bins) or None if not numerical."""
        for pat in self.HP_PATTERNS:
            if pat in name:
                return ("hp", 0.0, 1.0, self.hp_bins)
        for pat in self.STAT_PATTERNS:
            if pat in name:
                return ("stat", 0.0, 600.0, self.stat_bins)
        for pat in self.POWER_PATTERNS:
            if pat in name:
                return ("power", 0.0, 250.0, self.power_bins)
        for pat in self.DAMAGE_PATTERNS:
            if pat in name:
                return ("damage", 0.0, 600.0, self.damage_bins)
        for pat in self.TURN_PATTERNS:
            if name == pat:  # exact match to avoid matching e.g. "turn_head"
                return ("turn", 0.0, 40.0, self.turn_bins)
        for pat in self.RATING_PATTERNS:
            if name == pat:  # exact match
                return ("rating", 0.0, 2000.0, self.rating_bins)
        return None

    def _embed_dim_for_entry(self, entry: Tuple[int, str, float, float, int]) -> int:
        """Return the embedding dimension for a classified feature entry."""
        bank_name = entry[1]
        return self._embed_dim_for_bank(bank_name)

    def _embed_dim_for_bank(self, bank_name: str) -> int:
        """Return the embedding dimension for a given bank."""
        if bank_name == "damage":
            return self.damage_embed_dim
        elif bank_name == "turn":
            return self.turn_embed_dim
        elif bank_name == "rating":
            return self.rating_embed_dim
        else:
            return self.embed_dim

    def _get_bank(self, bank_name: str) -> torch.nn.Embedding:
        if bank_name == "hp":
            return self.hp_bank
        elif bank_name == "stat":
            return self.stat_bank
        elif bank_name == "damage":
            return self.damage_bank
        elif bank_name == "turn":
            return self.turn_bank
        elif bank_name == "rating":
            return self.rating_bank
        else:
            return self.power_bank

    def embed_group(
        self,
        x: torch.Tensor,
        group_idx: int,
    ) -> torch.Tensor:
        """Replace numerical features in a single group with learned embeddings.

        Args:
            x: (batch, seq, group_size) — raw feature tensor for this group.
            group_idx: index into ``self._group_maps``.

        Returns:
            (batch, seq, new_group_size) with replaced features expanded.
        """
        gmap = self._group_maps[group_idx]
        if not gmap:
            return x  # No numerical features; pass through.

        batch, seq, _ = x.shape
        parts: List[torch.Tensor] = []
        prev_end = 0

        # Sort by local_idx so we iterate left-to-right.
        sorted_entries = sorted(gmap, key=lambda e: e[0])

        for local_idx, bank_name, min_val, max_val, n_bins in sorted_entries:
            # Keep non-numerical columns before this one.
            if local_idx > prev_end:
                parts.append(x[:, :, prev_end:local_idx])

            # Discretize: clamp to [min_val, max_val], scale to [0, n_bins].
            raw = x[:, :, local_idx]  # (batch, seq)
            # Treat sentinel -1 as 0 (absent features)
            clamped = raw.clamp(min=min_val, max=max_val)
            bucket = ((clamped - min_val) / (max_val - min_val) * n_bins).long()
            bucket = bucket.clamp(0, n_bins)  # safety

            bank = self._get_bank(bank_name)
            embedded = bank(bucket)  # (batch, seq, embed_dim)
            parts.append(embedded)

            prev_end = local_idx + 1

        # Append remaining columns after last replaced feature.
        if prev_end < x.shape[2]:
            parts.append(x[:, :, prev_end:])

        return torch.cat(parts, dim=-1)


class EntityIDEncoder(torch.nn.Module):
    """Replaces integer ID features (ability_id, item_id) with learned embeddings.

    Similar in API to ``NumberBankEncoder``, this module identifies which feature
    positions within each group hold entity IDs (by pattern-matching feature names)
    and replaces the single scalar ID with a dense embedding vector via
    ``nn.Embedding``.

    Entity IDs use -1 for absent/unknown pokemon and 0 for unknown entity of a
    known pokemon. Both are mapped to the padding index (0) during lookup.

    Usage: applied inside ``GroupedFeatureEncoder.forward()`` before the per-group
    linear encoder, alongside (and before) ``NumberBankEncoder``.
    """

    # Feature name patterns → entity type key
    ID_PATTERNS = {
        "ability_id": "ability",
        "item_id": "item",
        "species_id": "species",
        "move_id": "move",
    }

    def __init__(
        self,
        feature_names: List[str],
        group_sizes: List[int],
        num_abilities: int,
        num_items: int,
        num_species: int,
        num_moves: int,
        ability_embed_dim: int = 16,
        item_embed_dim: int = 16,
        species_embed_dim: int = 32,
        move_embed_dim: int = 16,
    ):
        super().__init__()
        self.ability_embed_dim = ability_embed_dim
        self.item_embed_dim = item_embed_dim
        self.species_embed_dim = species_embed_dim
        self.move_embed_dim = move_embed_dim

        # Embedding tables: index 0 = unknown/padding
        self.ability_emb = torch.nn.Embedding(
            num_abilities, ability_embed_dim, padding_idx=0
        )
        self.item_emb = torch.nn.Embedding(
            num_items, item_embed_dim, padding_idx=0
        )
        self.species_emb = torch.nn.Embedding(
            num_species, species_embed_dim, padding_idx=0
        )
        self.move_emb = torch.nn.Embedding(
            num_moves, move_embed_dim, padding_idx=0
        )

        # Build per-group maps: for each group, which positions are entity IDs
        # Each entry: (local_idx, entity_type)
        self._group_maps: List[List[Tuple[int, str]]] = []
        self._group_output_sizes: List[int] = []

        offset = 0
        for gsize in group_sizes:
            gmap: List[Tuple[int, str]] = []
            for local_idx in range(gsize):
                global_idx = offset + local_idx
                if global_idx >= len(feature_names):
                    break
                name = feature_names[global_idx]
                for pattern, entity_type in self.ID_PATTERNS.items():
                    if name.endswith(pattern):
                        gmap.append((local_idx, entity_type))
                        break
            self._group_maps.append(gmap)
            # Each replaced scalar becomes embed_dim features
            n_replaced = len(gmap)
            embed_expansion = sum(
                self._embed_dim_for(etype) for _, etype in gmap
            )
            self._group_output_sizes.append(gsize - n_replaced + embed_expansion)
            offset += gsize

    def _embed_dim_for(self, entity_type: str) -> int:
        if entity_type == "ability":
            return self.ability_embed_dim
        elif entity_type == "item":
            return self.item_embed_dim
        elif entity_type == "species":
            return self.species_embed_dim
        else:  # move
            return self.move_embed_dim

    def _get_embedding(self, entity_type: str) -> torch.nn.Embedding:
        if entity_type == "ability":
            return self.ability_emb
        elif entity_type == "item":
            return self.item_emb
        elif entity_type == "species":
            return self.species_emb
        else:  # move
            return self.move_emb

    @property
    def group_output_sizes(self) -> List[int]:
        """Per-group feature dimension after entity-ID expansion."""
        return self._group_output_sizes

    def embed_group(self, x: torch.Tensor, group_idx: int) -> torch.Tensor:
        """Replace entity ID features in a single group with learned embeddings.

        Args:
            x: (batch, seq, group_size) — raw feature tensor for this group.
            group_idx: index into ``self._group_maps``.

        Returns:
            (batch, seq, new_group_size) with ID features replaced by embeddings.
        """
        gmap = self._group_maps[group_idx]
        if not gmap:
            return x  # No entity ID features; pass through.

        parts: List[torch.Tensor] = []
        prev_end = 0

        sorted_entries = sorted(gmap, key=lambda e: e[0])

        for local_idx, entity_type in sorted_entries:
            # Keep non-ID columns before this one.
            if local_idx > prev_end:
                parts.append(x[:, :, prev_end:local_idx])

            # Extract ID, clamp sentinel -1 to 0 (the padding index)
            raw_id = x[:, :, local_idx].long().clamp(min=0)
            emb_layer = self._get_embedding(entity_type)
            embedded = emb_layer(raw_id)  # (batch, seq, embed_dim)
            parts.append(embedded)

            prev_end = local_idx + 1

        # Append remaining columns after last replaced feature.
        if prev_end < x.shape[2]:
            parts.append(x[:, :, prev_end:])

        return torch.cat(parts, dim=-1)


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
        early_layers: list,
        late_layers: list,
        embedder: Embedder,
        lstm_layers: int = 2,
        lstm_hidden_size: int = 512,
        num_actions: int = MDBO.action_space(),
        num_teampreview_actions: int = MDBO.teampreview_space(),
        max_seq_len: int = 40,
        dropout: float = 0.1,
        early_attention_heads: int = 8,
        late_attention_heads: int = 8,
        grouped_encoder_hidden_dim: int = 128,
        grouped_encoder_aggregated_dim: int = 1024,
        pokemon_attention_heads: int = 2,
        teampreview_head_layers: Optional[list] = None,
        teampreview_head_dropout: float = 0.1,
        teampreview_attention_heads: int = 4,
        turn_head_layers: Optional[list] = None,
        num_value_bins: int = 51,
        value_min: float = -1.0,
        value_max: float = 1.0,
        # Grouped feature expansion hyperparameters (architecture choices, not data)
        number_bank_hp_bins: int = 100,
        number_bank_stat_bins: int = 600,
        number_bank_power_bins: int = 250,
        number_bank_embedding_dim: int = 16,
        number_bank_damage_bins: int = 600,
        number_bank_damage_embed_dim: int = 4,
        number_bank_turn_bins: int = 40,
        number_bank_turn_embed_dim: int = 16,
        number_bank_rating_bins: int = 100,
        number_bank_rating_embed_dim: int = 16,
        ability_embed_dim: int = 16,
        item_embed_dim: int = 16,
        species_embed_dim: int = 32,
        move_embed_dim: int = 16,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.hidden_size = early_layers[-1] if early_layers else embedder.embedding_size
        self.num_actions = num_actions
        self.num_teampreview_actions = num_teampreview_actions
        self.early_attention_heads = early_attention_heads
        self.late_attention_heads = late_attention_heads
        self.teampreview_head_layers = teampreview_head_layers or []
        self.turn_head_layers = turn_head_layers or []
        self.num_value_bins = num_value_bins

        # Feature encoding: either grouped or simple linear
        if embedder.feature_set != Embedder.SIMPLE:
            self.feature_encoder: Optional[GroupedFeatureEncoder] = GroupedFeatureEncoder(
                group_sizes=embedder.group_embedding_sizes,
                feature_names=embedder.feature_names,
                num_abilities=embedder.num_abilities,
                num_items=embedder.num_items,
                num_species=embedder.num_species,
                num_moves=embedder.num_moves,
                hidden_dim=grouped_encoder_hidden_dim,
                aggregated_dim=grouped_encoder_aggregated_dim,
                dropout=dropout,
                pokemon_attention_heads=pokemon_attention_heads,
                ability_embed_dim=ability_embed_dim,
                item_embed_dim=item_embed_dim,
                species_embed_dim=species_embed_dim,
                move_embed_dim=move_embed_dim,
                number_bank_hp_bins=number_bank_hp_bins,
                number_bank_stat_bins=number_bank_stat_bins,
                number_bank_power_bins=number_bank_power_bins,
                number_bank_embedding_dim=number_bank_embedding_dim,
                number_bank_damage_bins=number_bank_damage_bins,
                number_bank_damage_embed_dim=number_bank_damage_embed_dim,
                number_bank_turn_bins=number_bank_turn_bins,
                number_bank_turn_embed_dim=number_bank_turn_embed_dim,
                number_bank_rating_bins=number_bank_rating_bins,
                number_bank_rating_embed_dim=number_bank_rating_embed_dim,
            )
            # early_ff_stack input is aggregated_dim, NOT early_layers[0]
            early_ff_layers = []
            prev_size = grouped_encoder_aggregated_dim
            for h in early_layers:
                early_ff_layers.append(ResidualBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.early_ff_stack = (
                torch.nn.Sequential(*early_ff_layers)
                if early_ff_layers
                else torch.nn.Identity()
            )
        else:
            # Simple linear projection
            self.feature_encoder = None
            input_proj = torch.nn.Linear(embedder.embedding_size, early_layers[0])
            init_linear_layer(input_proj)
            self.input_proj = input_proj

            # early_ff_stack processes [early_layers[0] → early_layers[1] → ...]
            early_ff_layers = []
            prev_size = early_layers[0]
            for h in early_layers[1:]:
                early_ff_layers.append(ResidualBlock(prev_size, h, dropout=dropout))
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
            teampreview_ff_layers.append(
                ResidualBlock(prev_size, h, dropout=teampreview_head_dropout)
            )
            prev_size = h

        teampreview_output_size = (
            self.teampreview_head_layers[-1]
            if self.teampreview_head_layers
            else self.hidden_size
        )

        # Optional attention for teampreview head
        if teampreview_attention_heads > 0:
            self.teampreview_attn: Optional[torch.nn.MultiheadAttention] = (
                torch.nn.MultiheadAttention(
                    teampreview_output_size,
                    teampreview_attention_heads,
                    batch_first=True,
                    dropout=teampreview_head_dropout,
                )
            )
            self.teampreview_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(
                teampreview_output_size
            )
        else:
            self.teampreview_attn = None
            self.teampreview_ln = None

        self.teampreview_ff_stack = (
            torch.nn.Sequential(*teampreview_ff_layers)
            if teampreview_ff_layers
            else torch.nn.Identity()
        )

        # Teampreview action head
        self.teampreview_head = torch.nn.Linear(
            teampreview_output_size, num_teampreview_actions
        )
        torch.nn.init.xavier_normal_(self.teampreview_head.weight, gain=0.01)
        torch.nn.init.constant_(self.teampreview_head.bias, 0)

        # Early attention if enabled (heads > 0)
        if early_attention_heads > 0:
            self.early_attn: Optional[torch.nn.MultiheadAttention] = (
                torch.nn.MultiheadAttention(
                    self.hidden_size,
                    early_attention_heads,
                    batch_first=True,
                    dropout=dropout,
                )
            )
            self.early_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(
                self.hidden_size
            )
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
            self.skip_proj: Optional[torch.nn.Linear] = torch.nn.Linear(
                self.hidden_size, lstm_output_size
            )
            torch.nn.init.xavier_normal_(self.skip_proj.weight, gain=0.01)
            torch.nn.init.constant_(self.skip_proj.bias, 0)
        else:
            self.skip_proj = None

        # Late attention if enabled (operates on LSTM output size)
        if late_attention_heads > 0:
            self.late_attn: Optional[torch.nn.MultiheadAttention] = (
                torch.nn.MultiheadAttention(
                    lstm_output_size,
                    late_attention_heads,
                    batch_first=True,
                    dropout=dropout,
                )
            )
            self.late_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(
                lstm_output_size
            )
        else:
            self.late_attn = None
            self.late_ln = None

        # Build late feedforward stack with residual blocks
        late_ff_layers = []
        prev_size = lstm_output_size
        for h in late_layers:
            late_ff_layers.append(ResidualBlock(prev_size, h, dropout=dropout))
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
            turn_ff_layers.append(ResidualBlock(prev_size, h, dropout=dropout))
            prev_size = h

        turn_output_size = (
            self.turn_head_layers[-1] if self.turn_head_layers else output_size
        )

        self.turn_ff_stack = (
            torch.nn.Sequential(*turn_ff_layers) if turn_ff_layers else torch.nn.Identity()
        )

        # Final linear layer for turn actions
        self.turn_action_head = torch.nn.Linear(turn_output_size, num_actions)
        torch.nn.init.xavier_normal_(self.turn_action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.turn_action_head.bias, 0)

        # Win prediction head - C51 distributional value head
        # Outputs logits over num_value_bins bins spanning [value_min, value_max]
        # Expected value is computed as weighted sum of bin centers
        support = torch.linspace(value_min, value_max, num_value_bins)
        self.register_buffer("value_support", support)

        win_linear1 = torch.nn.Linear(output_size, 128)
        win_linear2 = torch.nn.Linear(128, num_value_bins)

        # Initialize with small values to prevent explosion
        torch.nn.init.xavier_normal_(win_linear1.weight, gain=0.01)
        torch.nn.init.constant_(win_linear1.bias, 0)
        torch.nn.init.xavier_normal_(win_linear2.weight, gain=0.01)
        torch.nn.init.constant_(win_linear2.bias, 0)

        self.win_head = torch.nn.Sequential(
            win_linear1,
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            win_linear2,
            # No activation — raw logits over bins
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through three-headed model.

        Returns:
            turn_action_logits: (batch, seq_len, 2025) - Turn action predictions
            teampreview_logits: (batch, seq_len, 90) - Teampreview action predictions
            win_values: (batch, seq_len) - Expected win value (from distributional head)
            win_dist_logits: (batch, seq_len, num_value_bins) - Raw distributional logits
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
                teampreview_features,
                teampreview_features,
                teampreview_features,
                key_padding_mask=attn_mask,
            )
            teampreview_features = self.teampreview_ln(teampreview_features + tp_attn_out)  # type: ignore

        teampreview_logits = self.teampreview_head(
            teampreview_features
        )  # (batch, seq, 90)

        # Early attention if enabled
        if self.early_attn is not None:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            attn_out, _ = self.early_attn(
                ff_out_early, ff_out_early, ff_out_early, key_padding_mask=attn_mask
            )
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
        win_dist_logits = self.win_head(out)  # (batch, seq_len, num_value_bins)
        win_probs = torch.softmax(win_dist_logits, dim=-1)
        win_values = (win_probs * self.value_support).sum(dim=-1)  # type: ignore[operator]  # (batch, seq_len)

        return turn_action_logits, teampreview_logits, win_values, win_dist_logits

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
    ]:
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
            win_values: (batch, seq_len) - Expected win value (from distributional head)
            win_dist_logits: (batch, seq_len, num_value_bins) - Raw distributional logits
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
                teampreview_features,
                teampreview_features,
                teampreview_features,
                key_padding_mask=attn_mask,
            )
            teampreview_features = self.teampreview_ln(teampreview_features + tp_attn_out)  # type: ignore

        teampreview_logits = self.teampreview_head(teampreview_features)

        # Early attention if enabled
        if self.early_attn is not None:
            attn_mask = ~mask.bool() if mask is not None else None
            attn_out, _ = self.early_attn(
                ff_out_early, ff_out_early, ff_out_early, key_padding_mask=attn_mask
            )
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
                dtype=x.dtype,
            )
            c_0 = torch.zeros(
                num_layers * num_directions,
                batch_size,
                self.lstm.hidden_size,
                device=x.device,
                dtype=x.dtype,
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
        win_dist_logits = self.win_head(out)  # (batch, seq, num_value_bins)
        win_probs = torch.softmax(win_dist_logits, dim=-1)
        win_values = (win_probs * self.value_support).sum(dim=-1)  # type: ignore[operator]  # (batch, seq)

        return turn_action_logits, teampreview_logits, win_values, win_dist_logits, next_hidden

    def predict(self, x: torch.Tensor, mask=None):
        with torch.no_grad():
            turn_action_logits, teampreview_logits, win_values, _ = self.forward(x, mask)
            turn_action_probs = torch.softmax(turn_action_logits, dim=-1)
            teampreview_probs = torch.softmax(teampreview_logits, dim=-1)
        return turn_action_probs, teampreview_probs, win_values


# ---------------------------------------------------------------------------
# Transformer architecture
# ---------------------------------------------------------------------------


class SinusoidalPositionalEncoding(torch.nn.Module):
    """Standard sinusoidal positional encoding for Transformer models."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


class TransformerThreeHeadedModel(torch.nn.Module):
    """Transformer-based three-headed model for VGC doubles.

    Replaces the bidirectional LSTM backbone of ``FlexibleThreeHeadedModel``
    with a Transformer encoder that uses **decision tokens** (learned query
    vectors) following the ps-ppo design.

    Architecture
    ============
    Input → (optional) GroupedFeatureEncoder → early_ff_stack
          → Prepend decision tokens: [ACTOR] [CRITIC] [FIELD]
          → Sinusoidal Positional Encoding
          → TransformerEncoder (N layers, causal mask)
          → Extract [ACTOR] → turn_action_head
          → Extract [CRITIC] → win_head (distributional)
          → teampreview branches before transformer (same as LSTM model)

    RL hidden-state management
    --------------------------
    Unlike the LSTM model which carries ``(h, c)`` hidden state, the
    Transformer variant stores the growing *context window* of previously
    encoded features.  ``forward_with_hidden`` accepts/returns this context
    tensor so the RL player can accumulate history across turns.
    """

    # Number of special decision tokens prepended to the sequence.
    NUM_DECISION_TOKENS = 3  # [ACTOR, CRITIC, FIELD]

    def __init__(
        self,
        early_layers: list,
        late_layers: list,
        embedder: Embedder,
        num_actions: int = MDBO.action_space(),
        num_teampreview_actions: int = MDBO.teampreview_space(),
        max_seq_len: int = 40,
        dropout: float = 0.1,
        # Grouped encoder parameters
        grouped_encoder_hidden_dim: int = 128,
        grouped_encoder_aggregated_dim: int = 1024,
        pokemon_attention_heads: int = 2,
        # Teampreview head
        teampreview_head_layers: Optional[list] = None,
        teampreview_head_dropout: float = 0.1,
        teampreview_attention_heads: int = 4,
        # Turn head
        turn_head_layers: Optional[list] = None,
        # Distributional value head
        num_value_bins: int = 51,
        value_min: float = -1.0,
        value_max: float = 1.0,
        # Grouped feature expansion hyperparameters (architecture choices, not data)
        number_bank_hp_bins: int = 100,
        number_bank_stat_bins: int = 600,
        number_bank_power_bins: int = 250,
        number_bank_embedding_dim: int = 16,
        number_bank_damage_bins: int = 600,
        number_bank_damage_embed_dim: int = 4,
        number_bank_turn_bins: int = 40,
        number_bank_turn_embed_dim: int = 16,
        number_bank_rating_bins: int = 100,
        number_bank_rating_embed_dim: int = 16,
        ability_embed_dim: int = 16,
        item_embed_dim: int = 16,
        species_embed_dim: int = 32,
        move_embed_dim: int = 16,
        # Transformer-specific
        transformer_layers: int = 6,
        transformer_heads: int = 16,
        transformer_ff_dim: int = 2048,
        transformer_dropout: float = 0.1,
        use_decision_tokens: bool = True,
        use_causal_mask: bool = True,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.early_layers = early_layers
        self.late_layers = late_layers
        self.hidden_size = early_layers[-1] if early_layers else embedder.embedding_size
        self.num_actions = num_actions
        self.num_teampreview_actions = num_teampreview_actions
        self.num_value_bins = num_value_bins
        self.teampreview_head_layers = teampreview_head_layers or []
        self.turn_head_layers = turn_head_layers or []
        self.use_decision_tokens = use_decision_tokens
        self.use_causal_mask = use_causal_mask

        # ---- Feature encoder (shared with LSTM model) ----
        if embedder.feature_set != Embedder.SIMPLE:
            self.feature_encoder: Optional[GroupedFeatureEncoder] = GroupedFeatureEncoder(
                group_sizes=embedder.group_embedding_sizes,
                feature_names=embedder.feature_names,
                num_abilities=embedder.num_abilities,
                num_items=embedder.num_items,
                num_species=embedder.num_species,
                num_moves=embedder.num_moves,
                hidden_dim=grouped_encoder_hidden_dim,
                aggregated_dim=grouped_encoder_aggregated_dim,
                dropout=dropout,
                pokemon_attention_heads=pokemon_attention_heads,
                ability_embed_dim=ability_embed_dim,
                item_embed_dim=item_embed_dim,
                species_embed_dim=species_embed_dim,
                move_embed_dim=move_embed_dim,
                number_bank_hp_bins=number_bank_hp_bins,
                number_bank_stat_bins=number_bank_stat_bins,
                number_bank_power_bins=number_bank_power_bins,
                number_bank_embedding_dim=number_bank_embedding_dim,
                number_bank_damage_bins=number_bank_damage_bins,
                number_bank_damage_embed_dim=number_bank_damage_embed_dim,
                number_bank_turn_bins=number_bank_turn_bins,
                number_bank_turn_embed_dim=number_bank_turn_embed_dim,
                number_bank_rating_bins=number_bank_rating_bins,
                number_bank_rating_embed_dim=number_bank_rating_embed_dim,
            )
            early_ff_layers: list = []
            prev_size = grouped_encoder_aggregated_dim
            for h in early_layers:
                early_ff_layers.append(ResidualBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.early_ff_stack = (
                torch.nn.Sequential(*early_ff_layers) if early_ff_layers else torch.nn.Identity()
            )
        else:
            self.feature_encoder = None
            input_proj = torch.nn.Linear(embedder.embedding_size, early_layers[0])
            init_linear_layer(input_proj)
            self.input_proj = input_proj

            early_ff_layers = []
            prev_size = early_layers[0]
            for h in early_layers[1:]:
                early_ff_layers.append(ResidualBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.early_ff_stack = (
                torch.nn.Sequential(*early_ff_layers) if early_ff_layers else torch.nn.Identity()
            )
        # ---- Teampreview head (branches before the backbone, same as LSTM) ----
        tp_ff_layers: list = []
        prev_size = self.hidden_size
        for h in self.teampreview_head_layers:
            tp_ff_layers.append(ResidualBlock(prev_size, h, dropout=teampreview_head_dropout))
            prev_size = h
        tp_output_size = self.teampreview_head_layers[-1] if self.teampreview_head_layers else self.hidden_size

        if teampreview_attention_heads > 0:
            self.teampreview_attn: Optional[torch.nn.MultiheadAttention] = (
                torch.nn.MultiheadAttention(tp_output_size, teampreview_attention_heads, batch_first=True, dropout=teampreview_head_dropout)
            )
            self.teampreview_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(tp_output_size)
        else:
            self.teampreview_attn = None
            self.teampreview_ln = None

        self.teampreview_ff_stack = (
            torch.nn.Sequential(*tp_ff_layers) if tp_ff_layers else torch.nn.Identity()
        )
        self.teampreview_head = torch.nn.Linear(tp_output_size, num_teampreview_actions)
        torch.nn.init.xavier_normal_(self.teampreview_head.weight, gain=0.01)
        torch.nn.init.constant_(self.teampreview_head.bias, 0)

        # ---- Decision tokens ----
        if use_decision_tokens:
            self.register_parameter(
                "actor_token",
                torch.nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02),
            )
            self.register_parameter(
                "critic_token",
                torch.nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02),
            )
            self.register_parameter(
                "field_token",
                torch.nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02),
            )
        else:
            self.register_parameter("actor_token", None)
            self.register_parameter("critic_token", None)
            self.register_parameter("field_token", None)

        # ---- Positional encoding (sinusoidal) ----
        # +3 for decision tokens; +max_seq_len for turns
        self.pos_encoder = SinusoidalPositionalEncoding(
            self.hidden_size, max_len=max_seq_len + self.NUM_DECISION_TOKENS + 1
        )

        # ---- Transformer encoder backbone ----
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=transformer_dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )

        # ---- Late feedforward stack (processes transformer output) ----
        late_ff_layers_list: list = []
        prev_size = self.hidden_size
        for h in late_layers:
            late_ff_layers_list.append(ResidualBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.late_ff_stack = (
            torch.nn.Sequential(*late_ff_layers_list) if late_ff_layers_list else torch.nn.Identity()
        )
        output_size = late_layers[-1] if late_layers else self.hidden_size

        # ---- Turn action head ----
        turn_ff_layers_list: list = []
        prev_size = output_size
        for h in self.turn_head_layers:
            turn_ff_layers_list.append(ResidualBlock(prev_size, h, dropout=dropout))
            prev_size = h
        turn_output_size = self.turn_head_layers[-1] if self.turn_head_layers else output_size
        self.turn_ff_stack = (
            torch.nn.Sequential(*turn_ff_layers_list) if turn_ff_layers_list else torch.nn.Identity()
        )
        self.turn_action_head = torch.nn.Linear(turn_output_size, num_actions)
        torch.nn.init.xavier_normal_(self.turn_action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.turn_action_head.bias, 0)

        # ---- Win prediction head (distributional) ----
        support = torch.linspace(value_min, value_max, num_value_bins)
        self.register_buffer("value_support", support)

        win_linear1 = torch.nn.Linear(output_size, 128)
        win_linear2 = torch.nn.Linear(128, num_value_bins)
        torch.nn.init.xavier_normal_(win_linear1.weight, gain=0.01)
        torch.nn.init.constant_(win_linear1.bias, 0)
        torch.nn.init.xavier_normal_(win_linear2.weight, gain=0.01)
        torch.nn.init.constant_(win_linear2.bias, 0)
        self.win_head = torch.nn.Sequential(
            win_linear1,
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            win_linear2,
        )

    def _build_causal_mask(self, total_len: int, device: torch.device) -> torch.Tensor:
        """Build causal attention mask for TransformerEncoder.

        Decision tokens (first 3) can attend to all turns.
        Each turn can attend to decision tokens and all *prior* turns (causal).

        Returns float mask where ``-inf`` means "cannot attend".
        """
        n_dt = self.NUM_DECISION_TOKENS if self.use_decision_tokens else 0
        mask = torch.zeros(total_len, total_len, device=device)

        if self.use_causal_mask and total_len > n_dt:
            # Turns portion: causal among themselves
            turn_len = total_len - n_dt
            # Use triu on a -inf-filled matrix so the lower triangle is set to 0
            # by triu (NOT via multiplication, since 0 * -inf = NaN).
            mask[n_dt:, n_dt:] = torch.triu(
                torch.full((turn_len, turn_len), float("-inf"), device=device),
                diagonal=1,
            )

        return mask

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run feature encoder + early ff stack."""
        if self.feature_encoder is not None:
            return self.early_ff_stack(self.feature_encoder(x))
        else:
            return self.early_ff_stack(self.input_proj(x))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full-sequence forward for supervised training (batch, seq, feat).

        Returns:
            turn_action_logits, teampreview_logits, win_values, win_dist_logits
        """
        batch_size, seq_len, _ = x.shape
        encoded = self._encode_features(x)  # (B, S, H)

        # Teampreview head (pre-backbone, detached to stop gradient flow to encoder)
        tp_detached = encoded.detach()
        tp_feat = self.teampreview_ff_stack(tp_detached)
        if self.teampreview_attn is not None:
            attn_mask = ~mask.bool() if mask is not None else None
            tp_ao, _ = self.teampreview_attn(tp_feat, tp_feat, tp_feat, key_padding_mask=attn_mask)
            tp_feat = self.teampreview_ln(tp_feat + tp_ao)  # type: ignore
        teampreview_logits = self.teampreview_head(tp_feat)

        # Prepend decision tokens
        if self.use_decision_tokens:
            dt = torch.cat(
                [
                    self.actor_token.expand(batch_size, -1, -1),  # type: ignore
                    self.critic_token.expand(batch_size, -1, -1),  # type: ignore
                    self.field_token.expand(batch_size, -1, -1),  # type: ignore
                ],
                dim=1,
            )  # (B, 3, H)
            full_seq = torch.cat([dt, encoded], dim=1)  # (B, 3+S, H)
        else:
            full_seq = encoded

        # Positional encoding
        full_seq = self.pos_encoder(full_seq)

        # Build attention mask
        attn_mask = (
            self._build_causal_mask(full_seq.size(1), x.device) if self.use_causal_mask else None
        )

        # Build key_padding_mask if a padding mask is provided
        src_key_padding_mask: Optional[torch.Tensor] = None
        if mask is not None:
            pad_mask = ~mask.bool()  # True = padding position
            if self.use_decision_tokens:
                # Decision tokens are never padded
                dt_pad = torch.zeros(batch_size, self.NUM_DECISION_TOKENS, device=x.device, dtype=torch.bool)
                src_key_padding_mask = torch.cat([dt_pad, pad_mask], dim=1)
            else:
                src_key_padding_mask = pad_mask

        # Transformer
        t_out = self.transformer(
            full_seq,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        # Extract outputs
        if self.use_decision_tokens:
            actor_out = t_out[:, 0:1, :].expand(-1, seq_len, -1)  # broadcast
            critic_out = t_out[:, 1:2, :].expand(-1, seq_len, -1)
            # Also use the last-turn representation for per-step outputs
            turn_out = t_out[:, self.NUM_DECISION_TOKENS:, :]  # (B, S, H)
        else:
            actor_out = t_out
            critic_out = t_out
            turn_out = t_out

        # Late feedforward + heads
        out_turn = self.late_ff_stack(turn_out if self.use_decision_tokens else actor_out)
        turn_features = self.turn_ff_stack(out_turn)
        turn_action_logits = self.turn_action_head(turn_features)

        out_critic = self.late_ff_stack(critic_out)
        win_dist_logits = self.win_head(out_critic)
        win_probs = torch.softmax(win_dist_logits, dim=-1)
        win_values = (win_probs * self.value_support).sum(dim=-1)  # type: ignore[operator]

        return turn_action_logits, teampreview_logits, win_values, win_dist_logits

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online RL forward with context accumulation.

        Unlike the LSTM model which uses ``(h, c)`` hidden state, the Transformer
        stores the *full context window* of previously encoded features.

        Args:
            x: Current turn features ``(batch, 1, input_size)``.
            hidden_state: Previous context ``(batch, T-1, hidden_size)`` or ``None``
                for the first turn.  **NOTE**: this is a single tensor, not a tuple.
            mask: Optional padding mask.

        Returns:
            turn_action_logits: ``(batch, 1, num_actions)``
            teampreview_logits: ``(batch, 1, num_tp_actions)``
            win_values: ``(batch, 1)``
            win_dist_logits: ``(batch, 1, num_value_bins)``
            next_context: ``(batch, T, hidden_size)`` — updated context for next call.
        """
        batch_size = x.size(0)
        encoded = self._encode_features(x)  # (B, 1, H)

        # Teampreview head (pre-backbone, detached to stop gradient flow to encoder)
        tp_detached = encoded.detach()
        tp_feat = self.teampreview_ff_stack(tp_detached)
        if self.teampreview_attn is not None:
            tp_ao, _ = self.teampreview_attn(tp_feat, tp_feat, tp_feat)
            tp_feat = self.teampreview_ln(tp_feat + tp_ao)  # type: ignore
        teampreview_logits = self.teampreview_head(tp_feat)

        # Build context: concatenate with previous encoded features
        if hidden_state is not None:
            context = torch.cat([hidden_state, encoded], dim=1)  # (B, T, H)
        else:
            context = encoded  # (B, 1, H)

        # Prepend decision tokens
        if self.use_decision_tokens:
            dt = torch.cat(
                [
                    self.actor_token.expand(batch_size, -1, -1),  # type: ignore
                    self.critic_token.expand(batch_size, -1, -1),  # type: ignore
                    self.field_token.expand(batch_size, -1, -1),  # type: ignore
                ],
                dim=1,
            )
            full_seq = torch.cat([dt, context], dim=1)
        else:
            full_seq = context

        # Positional encoding
        full_seq = self.pos_encoder(full_seq)

        # Causal mask
        attn_mask = (
            self._build_causal_mask(full_seq.size(1), x.device) if self.use_causal_mask else None
        )

        # Transformer
        t_out = self.transformer(full_seq, mask=attn_mask)

        # For online play, take the LAST turn position's output
        last_idx = -1
        if self.use_decision_tokens:
            actor_out = t_out[:, 0:1, :]  # [ACTOR] token
            critic_out = t_out[:, 1:2, :]  # [CRITIC] token
        else:
            actor_out = t_out[:, last_idx:, :]
            critic_out = t_out[:, last_idx:, :]

        # Late ff + heads
        out_turn = self.late_ff_stack(actor_out)
        turn_features = self.turn_ff_stack(out_turn)
        turn_action_logits = self.turn_action_head(turn_features)

        out_critic = self.late_ff_stack(critic_out)
        win_dist_logits = self.win_head(out_critic)
        win_probs = torch.softmax(win_dist_logits, dim=-1)
        win_values = (win_probs * self.value_support).sum(dim=-1)  # type: ignore[operator]

        return turn_action_logits, teampreview_logits, win_values, win_dist_logits, context

    def predict(self, x: torch.Tensor, mask=None):
        with torch.no_grad():
            turn_action_logits, teampreview_logits, win_values, _ = self.forward(x, mask)
            turn_action_probs = torch.softmax(turn_action_logits, dim=-1)
            teampreview_probs = torch.softmax(teampreview_logits, dim=-1)
        return turn_action_probs, teampreview_probs, win_values
