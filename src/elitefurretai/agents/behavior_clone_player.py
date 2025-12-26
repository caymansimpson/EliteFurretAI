import copy
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.encoder import MDBO
from elitefurretai.utils.battle_order_validator import is_valid_order


def init_linear_layer(layer: torch.nn.Linear) -> None:
    """Initialize a linear layer with Kaiming normal initialization."""
    torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
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

        ResBlock = ResidualBlock

        # Feature encoding: either grouped or simple linear
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
            # Simple linear projection
            self.feature_encoder = None
            input_proj = torch.nn.Linear(input_size, early_layers[0])
            init_linear_layer(input_proj)
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

        # Teampreview head (branches from early features before LSTM)
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

        # Early attention if enabled
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

        # Bidirectional LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm = torch.nn.LSTM(
            self.hidden_size,
            lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Skip connection projection
        lstm_output_size = lstm_hidden_size * 2
        if self.hidden_size != lstm_output_size:
            self.skip_proj: Optional[torch.nn.Linear] = torch.nn.Linear(self.hidden_size, lstm_output_size)
            torch.nn.init.xavier_normal_(self.skip_proj.weight, gain=0.01)
            torch.nn.init.constant_(self.skip_proj.bias, 0)
        else:
            self.skip_proj = None

        # Late attention if enabled
        if late_attention_heads > 0:
            self.late_attn: Optional[torch.nn.MultiheadAttention] = torch.nn.MultiheadAttention(
                lstm_output_size, late_attention_heads, batch_first=True, dropout=dropout
            )
            self.late_ln: Optional[torch.nn.LayerNorm] = torch.nn.LayerNorm(lstm_output_size)
        else:
            self.late_attn = None
            self.late_ln = None

        # Build late feedforward stack
        late_ff_layers = []
        prev_size = lstm_output_size
        for h in late_layers:
            late_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.late_ff_stack = (
            torch.nn.Sequential(*late_ff_layers) if late_ff_layers else torch.nn.Identity()
        )

        output_size = late_layers[-1] if late_layers else lstm_output_size

        # Turn head with optional deeper layers
        self.turn_head_layers = turn_head_layers or []

        turn_ff_layers = []
        prev_size = output_size
        for h in self.turn_head_layers:
            turn_ff_layers.append(ResBlock(prev_size, h, dropout=dropout))
            prev_size = h
        self.turn_ff_stack = (
            torch.nn.Sequential(*turn_ff_layers) if turn_ff_layers else torch.nn.Identity()
        )

        turn_output_size = self.turn_head_layers[-1] if self.turn_head_layers else output_size

        # Turn action head
        self.turn_action_head = torch.nn.Linear(turn_output_size, num_actions)
        torch.nn.init.xavier_normal_(self.turn_action_head.weight, gain=0.01)
        torch.nn.init.constant_(self.turn_action_head.bias, 0)

        # Win prediction head
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
            ff_out_early = self.feature_encoder(x)
        else:
            ff_out_early = self.input_proj(x)

        # Early feedforward stack
        ff_out_early = self.early_ff_stack(ff_out_early)

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

        teampreview_logits = self.teampreview_head(teampreview_features)

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

        # Turn head with optional deeper layers
        turn_features = self.turn_ff_stack(out)
        turn_action_logits = self.turn_action_head(turn_features)

        # Win prediction head
        win_logits = self.win_head(out).squeeze(-1)

        return turn_action_logits, teampreview_logits, win_logits

    def forward_with_hidden(
        self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass that supports passing and returning hidden states for RL.
        """
        batch_size, seq_len, _ = x.shape

        # Feature encoding
        if self.feature_encoder is not None:
            ff_out_early = self.feature_encoder(x)
        else:
            ff_out_early = self.input_proj(x)

        # Early feedforward stack
        ff_out_early = self.early_ff_stack(ff_out_early)

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

        teampreview_logits = self.teampreview_head(teampreview_features)

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
            positions = torch.clamp(positions, max=self.max_seq_len - 1)

        x_pos = ff_out_early + self.pos_embedding(positions)

        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=x.device)

        # LSTM
        if hidden is not None:
            lstm_out, next_hidden = self.lstm(x_pos, hidden)
        else:
            lstm_out, next_hidden = self.lstm(x_pos)

        # Skip connection from early features
        if self.skip_proj is not None:
            skip_out = self.skip_proj(ff_out_early)
            lstm_out = lstm_out + skip_out

        # Late attention if enabled
        if self.late_attn is not None:
            if mask is None:
                attn_mask = None
            else:
                attn_mask = ~mask.bool()
            attn_out, _ = self.late_attn(lstm_out, lstm_out, lstm_out, key_padding_mask=attn_mask)
            lstm_out = self.late_ln(lstm_out + attn_out)  # type: ignore

        # Late feedforward stack
        ff_out_late = self.late_ff_stack(lstm_out)

        # Turn head
        turn_features = self.turn_ff_stack(ff_out_late)
        turn_action_logits = self.turn_action_head(turn_features)

        # Win prediction head
        win_logits = self.win_head(ff_out_late).squeeze(-1)

        return turn_action_logits, teampreview_logits, win_logits, next_hidden

    def predict(
        self, x: torch.Tensor, mask=None
    ):
        with torch.no_grad():
            turn_action_logits, teampreview_logits, win_logits = self.forward(x, mask)
            turn_action_probs = torch.softmax(turn_action_logits, dim=-1)
            teampreview_probs = torch.softmax(teampreview_logits, dim=-1)
        return turn_action_probs, teampreview_probs, win_logits


class BCPlayer(Player):
    def __init__(
        self,
        teampreview_model_filepath: str,
        action_model_filepath: str,
        win_model_filepath: str,
        battle_format: str = "gen9vgc2023regulationc",
        probabilistic=True,
        device: str = "cpu",
        verbose: bool = False,
        **kwargs,
    ):
        # pull in all player information manually
        if verbose:
            print("[BCPlayer] Initializing player...")
        super().__init__(**kwargs, battle_format=battle_format)

        if verbose:
            print(f"[BCPlayer] Creating embedder for format: {battle_format}")
        self._embedder = Embedder(
            format=battle_format, feature_set=Embedder.FULL, omniscient=False
        )
        self._probabilistic = probabilistic
        self._trajectories: Dict[str, list] = {}
        self._device = device
        self._verbose = verbose

        # Load three separate models for teampreview, action, and win prediction
        # Each model has its config embedded in the .pt file
        if verbose:
            print(f"[BCPlayer] Loading teampreview model from: {teampreview_model_filepath}")
        self.teampreview_model, self.teampreview_config = self._load_model(teampreview_model_filepath, device)

        if verbose:
            print(f"[BCPlayer] Loading action model from: {action_model_filepath}")
        self.action_model, self.action_config = self._load_model(action_model_filepath, device)

        if verbose:
            print(f"[BCPlayer] Loading win prediction model from: {win_model_filepath}")
        self.win_model, self.win_config = self._load_model(win_model_filepath, device)

        if verbose:
            print("[BCPlayer] Initialization complete!")

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

    def _load_model(self, filepath: str, device: str = "cpu") -> Tuple[FlexibleThreeHeadedModel, Dict[str, Any]]:
        """
        Load model from new format with embedded config.

        Args:
            filepath: Path to model checkpoint
            device: Device to load model on (overrides config device)

        Returns:
            model: Loaded FlexibleThreeHeadedModel
            config: Full config dict from checkpoint
        """
        # Load checkpoint (expects {'model_state_dict': ..., 'config': ...})
        if self._verbose:
            print("  Loading checkpoint from disk...")
        checkpoint = torch.load(filepath, map_location=device)

        if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint or 'config' not in checkpoint:
            raise ValueError(
                f"Model file {filepath} is in old format (state_dict only). "
                f"Please migrate using scripts/prepare/migrate_model_configs.py"
            )

        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']

        # Build model from config
        if self._verbose:
            print("  Building model architecture...")
        model = FlexibleThreeHeadedModel(
            input_size=self._embedder.embedding_size,
            early_layers=config["early_layers"],
            late_layers=config["late_layers"],
            lstm_layers=config.get("lstm_layers", 2),
            lstm_hidden_size=config.get("lstm_hidden_size", 512),
            dropout=config.get("dropout", 0.1),
            gated_residuals=config.get("gated_residuals", False),
            early_attention_heads=config.get("early_attention_heads", 8),
            late_attention_heads=config.get("late_attention_heads", 8),
            use_grouped_encoder=config.get("use_grouped_encoder", False),
            group_sizes=self._embedder.group_embedding_sizes if config.get("use_grouped_encoder", False) else None,
            grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
            grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
            pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
            teampreview_head_layers=config.get("teampreview_head_layers", []),
            teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
            teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
            turn_head_layers=config.get("turn_head_layers", []),
            num_actions=MDBO.action_space(),
            num_teampreview_actions=MDBO.teampreview_space(),
            max_seq_len=17,
        ).to(device)

        if self._verbose:
            print("  Loading model weights...")
        model.load_state_dict(state_dict)
        model.eval()

        if self._verbose:
            print(f"  Model loaded successfully on device: {device}")

        return model, config

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
                await self.send_message("skill issue", battle_tag)

            # Check for dramatic negative swing (was winning, now losing)
            elif prev_advantage > self._win_advantage_threshold and current_advantage < -self._win_advantage_threshold:
                await self.send_message("misclick", battle_tag)

        # Update tracked advantage
        self._last_win_advantage[battle_tag] = current_advantage

    def predict(self, traj: torch.Tensor, battle: DoubleBattle, action_type: Optional[str] = None) -> Tuple[List[BattleOrder], List[float]]:
        """
        Given a trajectory tensor and battle, returns lists of valid actions and their probabilities
        for the last state in the trajectory.

        Args:
            traj: Trajectory tensor of shape (batch, seq_len, embed_dim)
            battle: Current battle state
            action_type: Optional action type (TEAMPREVIEW/TURN/FORCE_SWITCH). If None, inferred from battle state.

        Returns:
            Tuple of (actions, probabilities) where actions are BattleOrder objects (or strings for teampreview)
        """
        # Use appropriate model based on battle phase
        if action_type is None:
            # Infer action type from battle state
            if battle.teampreview:
                action_type = MDBO.TEAMPREVIEW
            elif any(battle.force_switch):
                action_type = MDBO.FORCE_SWITCH
            else:
                action_type = MDBO.TURN

        if action_type == MDBO.TEAMPREVIEW:
            model = self.teampreview_model
            max_actions = MDBO.teampreview_space()
        else:
            model = self.action_model
            max_actions = MDBO.action_space()

        # Truncate trajectory to model's max sequence length
        traj = traj[:, -model.max_seq_len :, :]  # type: ignore
        model.eval()
        with torch.no_grad():
            # Forward pass: get logits for all steps in the trajectory
            turn_action_logits, teampreview_logits, win_logits = model(traj)

            if turn_action_logits.dim() == 3:
                # Remove batch dimension if present
                turn_action_logits = turn_action_logits.squeeze(0)
                teampreview_logits = teampreview_logits.squeeze(0)
                win_logits = win_logits.squeeze(0)

            # Use appropriate head based on battle phase
            if battle.teampreview:
                last_logits = teampreview_logits[-1]  # shape: (90,)
            else:
                last_logits = turn_action_logits[-1]  # shape: (2025,)

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
                valid_count = 0
                for i in range(last_logits.size(0)):
                    try:
                        dbo = MDBO.from_int(i, action_type).to_double_battle_order(battle)
                        if is_valid_order(dbo, battle):  # type: ignore
                            mask[i] = 1
                            valid_count += 1
                    except Exception:
                        continue

            # Mask out invalid actions
            masked_logits = last_logits.masked_fill(~mask, float("-inf"))

            # Softmax over valid actions
            probs = torch.softmax(masked_logits, dim=-1)

            # Build output lists
            # CRITICAL: Only process actions that passed validation (are in the mask)
            actions = []
            probabilities = []
            mask_cpu = mask.cpu().numpy()
            probs_cpu = probs.cpu().numpy()

            for i in range(len(probs_cpu)):
                # Skip actions that weren't validated or have zero probability
                if not mask_cpu[i] or probs_cpu[i] <= 0 or i >= max_actions:
                    continue

                mdbo = MDBO.from_int(i, type=action_type)
                if battle.teampreview:
                    # For teampreview, store the message string
                    actions.append(mdbo.message)  # type: ignore
                    probabilities.append(float(probs_cpu[i]))
                else:
                    # For turn actions, convert to DoubleBattleOrder
                    try:
                        order = mdbo.to_double_battle_order(battle)
                        actions.append(order)
                        probabilities.append(float(probs_cpu[i]))
                    except Exception as e:
                        # This shouldn't happen since we already validated
                        print(f"[BCPlayer.predict] WARNING: Failed to convert validated action {i}: {e}")
                        continue

            return actions, probabilities  # type: ignore

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

        # Don't make choices if battle is over
        if battle.finished:
            return DefaultBattleOrder()

        # Embed and store the battle state
        state_vec = self.embed_battle_state(battle)
        if battle.battle_tag not in self._trajectories:
            self._trajectories[battle.battle_tag] = []
        self._trajectories[battle.battle_tag].append(state_vec)

        # Get model prediction based on the battle state
        actions, probabilities = self.predict(
            torch.Tensor(self._trajectories[battle.battle_tag]).unsqueeze(0), battle
        )

        if len(actions) == 0:
            print("WARNING in BCPlayer.choose_move: No valid actions available, returning random move.")
            return DefaultBattleOrder()

        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()  # Ensure sum to 1

        # If probabilistic, sample a move proportional to the softmax; otherwise, choose the best move
        if self._probabilistic:
            choice_idx = np.random.choice(len(actions), p=probabilities)
        else:
            choice_idx = int(np.argmax(probabilities))

        chosen_move = actions[choice_idx]

        # For teampreview, chosen_move is already a string message; return it directly
        # For turn actions, chosen_move is a DoubleBattleOrder object
        return chosen_move  # type: ignore

    def teampreview(self, battle: AbstractBattle) -> str:
        assert battle.player_role
        choice = self.choose_move(battle)

        # If it's already a string, use it; otherwise get .message
        message = choice if isinstance(choice, str) else choice.message

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
