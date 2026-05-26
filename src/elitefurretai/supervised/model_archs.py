# -*- coding: utf-8 -*-
"""Supervised model architecture for teampreview, turn action, and win prediction.

The active architecture is TransformerThreeHeadedModel: a Transformer encoder
over per-step embeddings (produced by GroupedFeatureEncoder) with decision
tokens and three output heads (teampreview, turn action, win).
"""

import math
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch

from elitefurretai.etl import MDBO
from elitefurretai.etl.embedder import Embedder


class _ValueTrunkGradScale(torch.autograd.Function):
    """Identity forward; scales gradient by `scale` on the backward pass.

    Inserted on the value-head path between the shared `late_ff_stack` and
    the value-specific `value_ff_stack`. Gradient flowing further upstream
    (into `late_ff_stack` parameters from this call, and the shared trunk
    via `t_out`) is multiplied by `scale`; the value-specific layers and
    `win_head` see full-magnitude gradient because they are downstream.
    Set `scale=1.0` to disable. Configured via
    `architecture.value_to_trunk_grad_scale` in YAML; see
    `planning/stage2/2026-05-16-22-30-value-grad-scale-and-mean-kl.md`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:  # type: ignore[override]
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore[override]
        return grad_output * ctx.scale, None


def scale_value_trunk_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Functional wrapper around `_ValueTrunkGradScale`. No-op when scale == 1.0."""
    if scale == 1.0:
        return x
    return _ValueTrunkGradScale.apply(x, scale)  # type: ignore[return-value]


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

        # Precompute per-group layouts so _dual_expand can run as a small number
        # of batched embedding lookups + scatter assignments instead of a Python
        # loop over individual positions.
        self._build_dual_expand_layouts()
        # Sanity-check that the precomputed output sizes match the path the rest
        # of __init__ uses to size encoders.
        for gi, eff in enumerate(effective_sizes):
            assert self._dual_expand_layouts[gi]["output_size"] == eff, (
                f"_dual_expand layout output size mismatch at group {gi}: "
                f"layout={self._dual_expand_layouts[gi]['output_size']} eff={eff}"
            )

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
        for group_idx, (encoder, size) in enumerate(zip(self.encoders, self.group_sizes)):
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

    def _build_dual_expand_layouts(self) -> None:
        """Precompute per-group layouts for vectorized _dual_expand.

        For each group, records: passthrough source/destination spans, plus
        per-entity-type and per-bank source-position / destination-index
        tensors. Source/destination tensors are registered as non-persistent
        buffers so they migrate with .to(device) without polluting state_dict.
        """
        self._dual_expand_layouts: List[Dict[str, Any]] = []

        for group_idx, gsize in enumerate(self.group_sizes):
            eid_entries = self.entity_id_encoder._group_maps[group_idx]
            nb_entries = self.number_bank._group_maps[group_idx]

            pos_info: Dict[int, Tuple[str, str, Optional[Tuple[float, float, int]]]] = {}
            for local_idx, entity_type in eid_entries:
                pos_info[local_idx] = ("eid", entity_type, None)
            for local_idx, bank_name, min_val, max_val, n_bins in nb_entries:
                pos_info[local_idx] = ("nb", bank_name, (min_val, max_val, n_bins))

            passthrough_spans: List[Tuple[int, int, int, int]] = []
            # etype -> list of (src_pos, dst_start, embed_dim)
            eid_dsts: Dict[str, List[Tuple[int, int, int]]] = {}
            # bank_name -> list of (src_pos, dst_start, embed_dim, min, max, n_bins)
            nb_dsts: Dict[str, List[Tuple[int, int, int, float, float, int]]] = {}

            prev_src_end = 0
            out_pos = 0
            for src_pos in sorted(pos_info.keys()):
                if src_pos > prev_src_end:
                    pass_len = src_pos - prev_src_end
                    passthrough_spans.append(
                        (prev_src_end, src_pos, out_pos, out_pos + pass_len)
                    )
                    out_pos += pass_len
                kind, key, params = pos_info[src_pos]
                if kind == "eid":
                    embed_dim = self.entity_id_encoder._embed_dim_for(key)
                    eid_dsts.setdefault(key, []).append((src_pos, out_pos, embed_dim))
                else:
                    assert params is not None
                    mn, mx, nb = params
                    embed_dim = self.number_bank._embed_dim_for_bank(key)
                    nb_dsts.setdefault(key, []).append(
                        (src_pos, out_pos, embed_dim, mn, mx, nb)
                    )
                out_pos += embed_dim
                prev_src_end = src_pos + 1
            if prev_src_end < gsize:
                pass_len = gsize - prev_src_end
                passthrough_spans.append(
                    (prev_src_end, gsize, out_pos, out_pos + pass_len)
                )
                out_pos += pass_len

            eid_layout: Dict[str, Dict[str, Any]] = {}
            for etype, items in eid_dsts.items():
                embed_dim = items[0][2]
                src_positions = torch.tensor([s[0] for s in items], dtype=torch.long)
                dst_idx_flat: List[int] = []
                for _, dst_start, ed in items:
                    dst_idx_flat.extend(range(dst_start, dst_start + ed))
                dst_indices = torch.tensor(dst_idx_flat, dtype=torch.long)
                src_buf = f"_dual_g{group_idx}_eid_{etype}_src"
                dst_buf = f"_dual_g{group_idx}_eid_{etype}_dst"
                self.register_buffer(src_buf, src_positions, persistent=False)
                self.register_buffer(dst_buf, dst_indices, persistent=False)
                eid_layout[etype] = {
                    "src_buf": src_buf,
                    "dst_buf": dst_buf,
                    "embed_dim": embed_dim,
                    "k": len(items),
                }

            nb_layout: Dict[str, Dict[str, Any]] = {}
            for bank_name, items in nb_dsts.items():
                embed_dim = items[0][2]
                mn = items[0][3]
                mx = items[0][4]
                nb = items[0][5]
                src_positions = torch.tensor([s[0] for s in items], dtype=torch.long)
                dst_idx_flat = []
                for _, dst_start, ed, _, _, _ in items:
                    dst_idx_flat.extend(range(dst_start, dst_start + ed))
                dst_indices = torch.tensor(dst_idx_flat, dtype=torch.long)
                src_buf = f"_dual_g{group_idx}_nb_{bank_name}_src"
                dst_buf = f"_dual_g{group_idx}_nb_{bank_name}_dst"
                self.register_buffer(src_buf, src_positions, persistent=False)
                self.register_buffer(dst_buf, dst_indices, persistent=False)
                nb_layout[bank_name] = {
                    "src_buf": src_buf,
                    "dst_buf": dst_buf,
                    "embed_dim": embed_dim,
                    "min_val": mn,
                    "max_val": mx,
                    "n_bins": nb,
                    "k": len(items),
                }

            self._dual_expand_layouts.append(
                {
                    "output_size": out_pos,
                    "passthrough": passthrough_spans,
                    "eid_layout": eid_layout,
                    "nb_layout": nb_layout,
                }
            )

    def _dual_expand(self, x: torch.Tensor, group_idx: int) -> torch.Tensor:
        """Vectorized expansion of entity IDs and number bank features.

        Replaces the per-position Python loop in _dual_expand_legacy with one
        batched embedding lookup per type/bank, scattered into a pre-allocated
        output tensor at the same positions the legacy version would have
        emitted. Output is bit-for-bit equivalent (modulo nondeterminism in
        the embedding ops themselves, which there is none of).
        """
        layout = self._dual_expand_layouts[group_idx]

        if not layout["eid_layout"] and not layout["nb_layout"]:
            return x

        B, T, _ = x.shape
        out = torch.empty(B, T, layout["output_size"], dtype=x.dtype, device=x.device)

        for src_s, src_e, dst_s, dst_e in layout["passthrough"]:
            out[:, :, dst_s:dst_e] = x[:, :, src_s:src_e]

        # Embedding lookups always return fp32 (embedding weights are fp32).
        # When the caller hands us bf16 states (battle_dataloader downcasts
        # for H2D bandwidth), `out` is bf16 and the raw `emb` is fp32 — eager
        # index_put rejects the mismatch. Cast to `out.dtype` so the function
        # is dtype-flexible in both eager and compiled paths. Compiled mode
        # previously hid this by silently promoting; we no longer rely on that.
        for etype, info in layout["eid_layout"].items():
            src_pos = getattr(self, info["src_buf"])  # (k,) long
            dst_idx = getattr(self, info["dst_buf"])  # (k * embed_dim,) long
            raw_ids = x[:, :, src_pos].long().clamp(min=0)  # (B, T, k)
            emb = self.entity_id_encoder._get_embedding(etype)(raw_ids)
            out[:, :, dst_idx] = emb.reshape(B, T, info["k"] * info["embed_dim"]).to(
                out.dtype
            )

        for bank_name, info in layout["nb_layout"].items():
            src_pos = getattr(self, info["src_buf"])
            dst_idx = getattr(self, info["dst_buf"])
            raw = x[:, :, src_pos]  # (B, T, k)
            mn = info["min_val"]
            mx = info["max_val"]
            nb_bins = info["n_bins"]
            clamped = raw.clamp(min=mn, max=mx)
            bucket = ((clamped - mn) / (mx - mn) * nb_bins).long().clamp(0, nb_bins)
            emb = self.number_bank._get_bank(bank_name)(bucket)
            out[:, :, dst_idx] = emb.reshape(B, T, info["k"] * info["embed_dim"]).to(
                out.dtype
            )

        return out

    def _dual_expand_legacy(self, x: torch.Tensor, group_idx: int) -> torch.Tensor:
        """Reference implementation kept solely for regression testing.

        Do not remove without porting equivalent coverage. _dual_expand must
        produce identical output for any valid input.
        """
        eid_map = {
            idx: etype for idx, etype in self.entity_id_encoder._group_maps[group_idx]
        }
        nb_map = {entry[0]: entry for entry in self.number_bank._group_maps[group_idx]}

        group_size = x.shape[2]
        parts: List[torch.Tensor] = []
        prev_end = 0

        all_positions = sorted(set(eid_map.keys()) | set(nb_map.keys()))

        for pos in all_positions:
            if pos > prev_end:
                parts.append(x[:, :, prev_end:pos])

            # Cast embedding outputs to x.dtype so torch.cat below doesn't
            # mix bf16 passthrough slices with fp32 embedding outputs.
            # Matches the parallel cast in _dual_expand so the two
            # implementations stay bit-identical under bf16 inputs.
            if pos in eid_map:
                raw_id = x[:, :, pos].long().clamp(min=0)
                emb_layer = self.entity_id_encoder._get_embedding(eid_map[pos])
                parts.append(emb_layer(raw_id).to(x.dtype))
            elif pos in nb_map:
                local_idx, bank_name, min_val, max_val, n_bins = nb_map[pos]
                raw = x[:, :, local_idx]
                clamped = raw.clamp(min=min_val, max=max_val)
                bucket = ((clamped - min_val) / (max_val - min_val) * n_bins).long()
                bucket = bucket.clamp(0, n_bins)
                bank = self.number_bank._get_bank(bank_name)
                parts.append(bank(bucket).to(x.dtype))

            prev_end = pos + 1

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
            embed_expansion = sum(self._embed_dim_for_entry(entry) for entry in gmap)
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
        self.item_emb = torch.nn.Embedding(num_items, item_embed_dim, padding_idx=0)
        self.species_emb = torch.nn.Embedding(
            num_species, species_embed_dim, padding_idx=0
        )
        self.move_emb = torch.nn.Embedding(num_moves, move_embed_dim, padding_idx=0)

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
            embed_expansion = sum(self._embed_dim_for(etype) for _, etype in gmap)
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

    Uses a Transformer encoder with **decision tokens** (learned query
    vectors) following the ps-ppo design.

    Architecture
    ============
    Input → (optional) GroupedFeatureEncoder → early_ff_stack
          → Prepend decision tokens: [ACTOR] [CRITIC] [FIELD]
          → Sinusoidal Positional Encoding
          → TransformerEncoder (N layers, causal mask)
          → Extract [ACTOR] → turn_action_head
          → Extract [CRITIC] → win_head (distributional)
          → teampreview branches before the transformer

    RL hidden-state management
    --------------------------
    The Transformer stores the growing *context window* of previously
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
        value_head_layers: Optional[list] = None,
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
        # Multiplier applied to the value-head gradient as it flows back into
        # the shared `late_ff_stack` and trunk. 1.0 = stock behavior; <1.0
        # dampens the value branch's contribution to shared-representation
        # gradients. Configured via `architecture.value_to_trunk_grad_scale`.
        # See planning/stage2/2026-05-16-22-30-value-grad-scale-and-mean-kl.md.
        value_to_trunk_grad_scale: float = 1.0,
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
        self.value_head_layers = value_head_layers or []
        self.use_decision_tokens = use_decision_tokens
        self.use_causal_mask = use_causal_mask
        self.value_to_trunk_grad_scale = value_to_trunk_grad_scale

        # ---- Feature encoder ----
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
                torch.nn.Sequential(*early_ff_layers)
                if early_ff_layers
                else torch.nn.Identity()
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
                torch.nn.Sequential(*early_ff_layers)
                if early_ff_layers
                else torch.nn.Identity()
            )
        # ---- Teampreview head (branches before the backbone) ----
        tp_ff_layers: list = []
        prev_size = self.hidden_size
        for h in self.teampreview_head_layers:
            tp_ff_layers.append(
                ResidualBlock(prev_size, h, dropout=teampreview_head_dropout)
            )
            prev_size = h
        tp_output_size = (
            self.teampreview_head_layers[-1]
            if self.teampreview_head_layers
            else self.hidden_size
        )

        if teampreview_attention_heads > 0:
            self.teampreview_attn: Optional[torch.nn.MultiheadAttention] = (
                torch.nn.MultiheadAttention(
                    tp_output_size,
                    teampreview_attention_heads,
                    batch_first=True,
                    dropout=teampreview_head_dropout,
                )
            )
            self.teampreview_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(
                tp_output_size
            )
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
            torch.nn.Sequential(*late_ff_layers_list)
            if late_ff_layers_list
            else torch.nn.Identity()
        )
        output_size = late_layers[-1] if late_layers else self.hidden_size

        # ---- Turn action head ----
        turn_ff_layers_list: list = []
        prev_size = output_size
        for h in self.turn_head_layers:
            turn_ff_layers_list.append(ResidualBlock(prev_size, h, dropout=dropout))
            prev_size = h
        turn_output_size = (
            self.turn_head_layers[-1] if self.turn_head_layers else output_size
        )
        self.turn_ff_stack = (
            torch.nn.Sequential(*turn_ff_layers_list)
            if turn_ff_layers_list
            else torch.nn.Identity()
        )
        self.turn_action_head = torch.nn.Linear(turn_output_size, num_actions)
        torch.nn.init.xavier_normal_(self.turn_action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.turn_action_head.bias, 0)

        # ---- Win prediction head (distributional) ----
        # Two modes:
        #   * value_head_layers empty (legacy): single 2-layer MLP win_head,
        #     output_size -> 128 -> num_value_bins. Shared depth with the
        #     policy head via late_ff_stack only.
        #   * value_head_layers non-empty (deep value head): a stack of
        #     ResidualBlocks built from value_head_layers feeds a final
        #     Linear projection to num_value_bins. Lets the value head do
        #     its own integrative computation without competing with policy
        #     for trunk representation capacity.
        support = torch.linspace(value_min, value_max, num_value_bins)
        self.register_buffer("value_support", support)

        if self.value_head_layers:
            value_ff_layers_list: list = []
            prev_size = output_size
            for h in self.value_head_layers:
                value_ff_layers_list.append(ResidualBlock(prev_size, h, dropout=dropout))
                prev_size = h
            self.value_ff_stack: torch.nn.Module = torch.nn.Sequential(
                *value_ff_layers_list
            )
            value_output_size = self.value_head_layers[-1]
            self.win_head: torch.nn.Module = torch.nn.Linear(
                value_output_size, num_value_bins
            )
            torch.nn.init.xavier_normal_(self.win_head.weight, gain=0.01)  # type: ignore[arg-type]
            torch.nn.init.constant_(self.win_head.bias, 0)  # type: ignore[arg-type]
        else:
            self.value_ff_stack = torch.nn.Identity()
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

        Returns bool mask where ``True`` means "cannot attend". Bool matches
        the dtype of `src_key_padding_mask` in the transformer call site;
        PyTorch deprecated mixing bool + float for these two masks.
        """
        n_dt = self.NUM_DECISION_TOKENS if self.use_decision_tokens else 0
        mask = torch.zeros(total_len, total_len, device=device, dtype=torch.bool)

        if self.use_causal_mask and total_len > n_dt:
            turn_len = total_len - n_dt
            mask[n_dt:, n_dt:] = torch.triu(
                torch.ones(turn_len, turn_len, device=device, dtype=torch.bool),
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
        # The training collate downcasts states to bf16 for H2D bandwidth
        # (battle_dataloader.py). Eager callers (evaluate, analyze, RL
        # inference) run with fp32 weights and reject mixed-dtype matmul.
        # Promote at the entry so every caller — eager or autocast'd — sees
        # the same contract. Under autocast(bf16) the next op will recast.
        if x.dtype != torch.float32:
            x = x.float()
        batch_size, seq_len, _ = x.shape
        encoded = self._encode_features(x)  # (B, S, H)

        # Teampreview head (pre-backbone, detached to stop gradient flow to encoder)
        tp_detached = encoded.detach()
        tp_feat = self.teampreview_ff_stack(tp_detached)
        if self.teampreview_attn is not None:
            attn_mask = ~mask.bool() if mask is not None else None
            tp_ao, _ = self.teampreview_attn(
                tp_feat, tp_feat, tp_feat, key_padding_mask=attn_mask
            )
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
            self._build_causal_mask(full_seq.size(1), x.device)
            if self.use_causal_mask
            else None
        )

        # Build key_padding_mask if a padding mask is provided
        src_key_padding_mask: Optional[torch.Tensor] = None
        if mask is not None:
            pad_mask = ~mask.bool()  # True = padding position
            if self.use_decision_tokens:
                # Decision tokens are never padded
                dt_pad = torch.zeros(
                    batch_size, self.NUM_DECISION_TOKENS, device=x.device, dtype=torch.bool
                )
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
            turn_out = t_out[:, self.NUM_DECISION_TOKENS :, :]  # (B, S, H)
        else:
            actor_out = t_out
            critic_out = t_out
            turn_out = t_out

        # Late feedforward + heads
        out_turn = self.late_ff_stack(turn_out if self.use_decision_tokens else actor_out)
        turn_features = self.turn_ff_stack(out_turn)
        turn_action_logits = self.turn_action_head(turn_features)

        out_critic = self.late_ff_stack(critic_out)
        out_critic = scale_value_trunk_gradient(out_critic, self.value_to_trunk_grad_scale)
        out_critic = self.value_ff_stack(out_critic)
        win_dist_logits = self.win_head(out_critic)
        win_probs = torch.softmax(win_dist_logits, dim=-1)
        win_values = (win_probs * self.value_support).sum(dim=-1)  # type: ignore[operator]

        return turn_action_logits, teampreview_logits, win_values, win_dist_logits

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        hidden_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Online RL forward with context accumulation.

        Stores the *full context window* of previously encoded features.

        Args:
            x: Current turn features ``(batch, 1, input_size)``.
            hidden_state: Previous context ``(batch, T-1, hidden_size)`` or ``None``
                for the first turn.  **NOTE**: this is a single tensor, not a tuple.
            mask: Optional padding mask.
            hidden_mask: Optional boolean mask for ``hidden_state`` with shape
                ``(batch, T-1)`` where ``True`` denotes a valid context position and
                ``False`` denotes padding. When omitted, all positions in
                ``hidden_state`` are treated as valid.

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
            if hidden_mask is not None:
                current_mask = torch.ones(
                    batch_size,
                    1,
                    device=x.device,
                    dtype=torch.bool,
                )
                context_mask = torch.cat([hidden_mask.to(torch.bool), current_mask], dim=1)
            else:
                context_mask = torch.ones(
                    batch_size,
                    context.size(1),
                    device=x.device,
                    dtype=torch.bool,
                )
        else:
            context = encoded  # (B, 1, H)
            context_mask = torch.ones(
                batch_size,
                1,
                device=x.device,
                dtype=torch.bool,
            )

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
            self._build_causal_mask(full_seq.size(1), x.device)
            if self.use_causal_mask
            else None
        )

        src_key_padding_mask: Optional[torch.Tensor] = None
        if self.use_decision_tokens:
            dt_pad = torch.zeros(
                batch_size,
                self.NUM_DECISION_TOKENS,
                device=x.device,
                dtype=torch.bool,
            )
            src_key_padding_mask = torch.cat([dt_pad, ~context_mask], dim=1)
        elif context_mask is not None:
            src_key_padding_mask = ~context_mask

        # Transformer
        t_out = self.transformer(
            full_seq,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

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
        out_critic = scale_value_trunk_gradient(out_critic, self.value_to_trunk_grad_scale)
        out_critic = self.value_ff_stack(out_critic)
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
