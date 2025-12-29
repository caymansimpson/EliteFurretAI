import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import BattleOrder, DefaultBattleOrder, Player

from elitefurretai.etl.battle_order_validator import is_valid_order
from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


class BCPlayer(Player):
    def __init__(
        self,
        teampreview_model_filepath: Optional[str] = None,
        action_model_filepath: Optional[str] = None,
        win_model_filepath: Optional[str] = None,
        unified_model_filepath: Optional[str] = None,
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

        # Validate model configuration
        has_unified = unified_model_filepath is not None
        has_separate = all(
            [
                teampreview_model_filepath is not None,
                action_model_filepath is not None,
                win_model_filepath is not None,
            ]
        )

        if not has_unified and not has_separate:
            raise ValueError(
                "Must provide either unified_model_filepath OR all three separate model filepaths "
                "(teampreview_model_filepath, action_model_filepath, win_model_filepath)"
            )

        if has_unified and has_separate:
            raise ValueError(
                "Cannot provide both unified_model_filepath and separate model filepaths. "
                "Choose one approach."
            )

        # Load models based on configuration
        if has_unified and isinstance(unified_model_filepath, str):
            # Load single unified model and point all three attributes to it
            if verbose:
                print(f"[BCPlayer] Loading unified model from: {unified_model_filepath}")
            unified_model, unified_config = self._load_model(
                unified_model_filepath, device
            )
            self.teampreview_model = unified_model
            self.action_model = unified_model
            self.win_model = unified_model
            self.teampreview_config = unified_config
            self.action_config = unified_config
            self.win_config = unified_config
            if verbose:
                print("[BCPlayer] Unified model loaded for all predictions")
        else:
            # Load three separate models for teampreview, action, and win prediction
            if verbose:
                print(
                    f"[BCPlayer] Loading teampreview model from: {teampreview_model_filepath}"
                )
            assert isinstance(teampreview_model_filepath, str)
            self.teampreview_model, self.teampreview_config = self._load_model(
                teampreview_model_filepath, device
            )

            if verbose:
                print(f"[BCPlayer] Loading action model from: {action_model_filepath}")
            assert isinstance(action_model_filepath, str)
            self.action_model, self.action_config = self._load_model(
                action_model_filepath, device
            )

            if verbose:
                print(
                    f"[BCPlayer] Loading win prediction model from: {win_model_filepath}"
                )
            assert isinstance(win_model_filepath, str)
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

    def _load_model(
        self, filepath: str, device: str = "cpu"
    ) -> Tuple[FlexibleThreeHeadedModel, Dict[str, Any]]:
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

        if (
            not isinstance(checkpoint, dict)
            or "model_state_dict" not in checkpoint
            or "config" not in checkpoint
        ):
            raise ValueError(
                f"Model file {filepath} is in old format (state_dict only). "
                f"Please migrate using scripts/prepare/migrate_model_configs.py"
            )

        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]

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
            group_sizes=(
                self._embedder.group_embedding_sizes
                if config.get("use_grouped_encoder", False)
                else None
            ),
            grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
            grouped_encoder_aggregated_dim=config.get(
                "grouped_encoder_aggregated_dim", 1024
            ),
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

    def predict_advantage(self, battle: DoubleBattle) -> float:
        """
        Predict win advantage for current battle state using stored trajectory.

        Args:
            battle: Current battle state

        Returns:
            Win advantage as scalar value in range [-1, 1] where:
            - 1.0 means 100% confidence you're winning
            - 0.0 means 50/50
            - -1.0 means 100% confidence you're losing
        """
        if (
            battle.battle_tag not in self._trajectories
            or len(self._trajectories[battle.battle_tag]) == 0
        ):
            # No trajectory yet, return neutral
            return 0.0

        # Build trajectory tensor
        traj = torch.Tensor(self._trajectories[battle.battle_tag]).unsqueeze(
            0
        )  # (1, seq_len, embed_dim)

        # Truncate to model's max sequence length
        traj = traj[:, -self.win_model.max_seq_len :, :]  # type: ignore

        self.win_model.eval()
        with torch.no_grad():
            # Forward pass through win model
            _, _, win_logits = self.win_model(traj)

            if win_logits.dim() == 2:
                # Remove batch dimension if present
                win_logits = win_logits.squeeze(0)

            # Get win advantage for last timestep
            # win_logits are already in [-1, 1] due to tanh activation in model
            win_advantage = float(win_logits[-1].item())

        return win_advantage

    async def _check_win_advantage_swing(
        self, battle: DoubleBattle, current_advantage: float
    ) -> None:
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
            if (
                prev_advantage < -self._win_advantage_threshold
                and current_advantage > self._win_advantage_threshold
            ):
                await self.send_message("skill issue", battle_tag)

            # Check for dramatic negative swing (was winning, now losing)
            elif (
                prev_advantage > self._win_advantage_threshold
                and current_advantage < -self._win_advantage_threshold
            ):
                await self.send_message("misclick", battle_tag)

        # Update tracked advantage
        self._last_win_advantage[battle_tag] = current_advantage

    def predict(
        self, traj: torch.Tensor, battle: DoubleBattle, action_type: Optional[str] = None
    ) -> Tuple[List[BattleOrder], List[float]]:
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
                        print(
                            f"[BCPlayer.predict] WARNING: Failed to convert validated action {i}: {e}"
                        )
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

        # Calculate and save win advantage
        win_advantage = self.predict_advantage(battle)
        self._last_win_advantage[battle.battle_tag] = win_advantage

        # Get model prediction based on the battle state
        actions, probabilities = self.predict(
            torch.Tensor(self._trajectories[battle.battle_tag]).unsqueeze(0), battle
        )

        if len(actions) == 0:
            print(
                "WARNING in BCPlayer.choose_move: No valid actions available, returning random move."
            )
            return DefaultBattleOrder()

        probabilities = np.array(probabilities)  # type: ignore
        probabilities = probabilities / probabilities.sum()  # type: ignore

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
        assert isinstance(battle, DoubleBattle)

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
