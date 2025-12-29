import asyncio
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
    DoubleBattleOrder,
)

from elitefurretai.etl.battle_order_validator import is_valid_order
from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.agent import RNaDAgent


class BatchInferencePlayer(Player):
    def __init__(
        self,
        model: RNaDAgent,
        device="cpu",
        batch_size=16,
        batch_timeout=0.01,
        probabilistic=True,
        trajectory_queue=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.probabilistic = probabilistic
        self.trajectory_queue = trajectory_queue
        self.embedder = Embedder(
            format=self.format, feature_set=Embedder.FULL, omniscient=False
        )
        self.queue: asyncio.Queue = asyncio.Queue()
        self.hidden_states: Dict[
            str, Tuple[torch.Tensor, torch.Tensor]
        ] = {}  # battle_tag -> (h, c)
        self._inference_task: Optional[asyncio.Task] = None

        # Trajectory storage
        # battle_tag -> list of dicts
        self.current_trajectories: Dict[str, List[Dict[str, Any]]] = {}
        self.completed_trajectories: List[
            Dict[str, Any]
        ] = []  # For debugging/inspection when no training queue

    async def start_inference_loop(self):
        self._inference_task = asyncio.create_task(self._inference_loop())

    async def _inference_loop(self):
        while True:
            batch: List[Any] = []
            futures: List[Any] = []
            battle_tags: List[str] = []
            is_tps: List[bool] = []
            masks: List[Any] = []
            # Collect batch
            try:
                # Wait for first item
                item = await self.queue.get()
                self._add_to_batch(batch, futures, battle_tags, is_tps, masks, item)

                # Collect more
                start_time = asyncio.get_event_loop().time()
                while len(batch) < self.batch_size:
                    timeout = self.batch_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        self._add_to_batch(
                            batch, futures, battle_tags, is_tps, masks, item
                        )
                    except asyncio.TimeoutError:
                        break
            except asyncio.CancelledError:
                break

            # Run inference
            if batch:
                await self._run_batch(batch, futures, battle_tags, is_tps, masks)

    def _add_to_batch(self, batch, futures, battle_tags, is_tps, masks, item):
        batch.append(item[0])
        futures.append(item[1])
        battle_tags.append(item[2])
        is_tps.append(item[3])
        masks.append(item[4])

    async def _run_batch(self, states, futures, battle_tags, is_tps, masks):
        # Prepare inputs
        states_tensor = (
            torch.tensor(np.array(states), dtype=torch.float32)
            .to(self.device)
            .unsqueeze(1)
        )  # (batch, 1, dim)

        # Get hidden states
        h_list = []
        c_list = []
        for tag in battle_tags:
            if tag not in self.hidden_states:
                h, c = self.model.get_initial_state(1, self.device)
                self.hidden_states[tag] = (h, c)
            h_list.append(self.hidden_states[tag][0])
            c_list.append(self.hidden_states[tag][1])

        h_batch = torch.cat(h_list, dim=1)
        c_batch = torch.cat(c_list, dim=1)
        hidden = (h_batch, c_batch)

        # Run model
        with torch.no_grad():
            turn_logits, tp_logits, values, next_hidden = self.model(states_tensor, hidden)

        # Update hidden states
        h_next, c_next = next_hidden
        for i, tag in enumerate(battle_tags):
            self.hidden_states[tag] = (h_next[:, i : i + 1, :], c_next[:, i : i + 1, :])

        # Process outputs
        turn_probs = torch.softmax(turn_logits, dim=-1).cpu().numpy()
        tp_probs = torch.softmax(tp_logits, dim=-1).cpu().numpy()
        values = values.cpu().numpy()

        for i, future in enumerate(futures):
            is_tp = is_tps[i]
            mask = masks[i]

            if is_tp:
                probs = tp_probs[i, 0]
                # TP mask? Usually not needed or handled by MDBO
                valid_actions = list(range(len(probs)))
            else:
                probs = turn_probs[i, 0]
                # Apply mask
                if mask is not None:
                    probs = probs * mask
                    if probs.sum() == 0:
                        probs = mask / mask.sum()  # Fallback to uniform over valid
                    else:
                        probs = probs / probs.sum()
                valid_actions = list(range(len(probs)))

            # Sample action
            if self.probabilistic:
                action = np.random.choice(valid_actions, p=probs)
            else:
                action = np.argmax(probs)

            log_prob = np.log(probs[action] + 1e-10)

            result = {
                "action": action,
                "log_prob": log_prob,
                "value": values[i, 0],
                "probs": probs,
            }
            future.set_result(result)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        """
        Override poke-env's request handler to route ALL decisions through our async pipeline.

        WHY WE OVERRIDE THIS:
        ---------------------
        poke-env's default _handle_battle_request does:
            if battle.teampreview:
                message = self.teampreview(battle)  # sync call, can't await!
            else:
                choice = self.choose_move(battle)   # supports await

        Problem: teampreview() is synchronous but we need async batched GPU inference.

        Solution: Override _handle_battle_request to call choose_move() for BOTH phases.
        This lets us use the same batched inference pipeline for teampreview and turns.
        """
        if maybe_default_order:
            # Handle potential invalid choice recovery (poke-env pattern)
            if random.random() < self.DEFAULT_CHOICE_CHANCE:
                message = self.choose_default_move().message
                await self.ps_client.send_message(message, battle.battle_tag)
                return

        # Route everything through choose_move for batched async inference
        choice = self.choose_move(battle)
        if asyncio.iscoroutine(choice):
            choice = await choice

        # Get the message string from the result
        if isinstance(choice, str):
            # Teampreview returns string directly (e.g., "/team 3412")
            message = choice
        elif hasattr(choice, "message"):
            # BattleOrder has a .message property
            message = choice.message
        else:
            message = str(choice)

        if message:
            await self.ps_client.send_message(message, battle.battle_tag)

    def choose_move(self, battle: AbstractBattle) -> Any:
        """
        Main decision method - handles BOTH teampreview and turn actions.

        This is called by our overridden _handle_battle_request for all phases.
        Returns an awaitable that resolves to either:
        - A string like "/team 3412" for teampreview
        - A BattleOrder for turn actions
        """
        return self._choose_move_async(battle)

    def teampreview(self, battle: AbstractBattle) -> str:
        """
        NOT USED - we override _handle_battle_request to bypass this.

        This method exists only as documentation. If called, something is wrong.
        """
        raise RuntimeError(
            "teampreview() should not be called - we override _handle_battle_request "
            "to route teampreview through choose_move() for async batched inference."
        )

    async def _choose_move_async(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return DefaultBattleOrder()

        # Embed state
        state = self.embedder.feature_dict_to_vector(self.embedder.embed(battle))

        # =====================================================================
        # ACTION MASKING: Prevent the model from selecting illegal moves
        # =====================================================================
        # VGC battles have two phases that require different action handling:
        #
        # 1. TEAMPREVIEW PHASE (battle.teampreview = True):
        #    - Player chooses 4 of their 6 Pokemon to bring to battle
        #    - All 90 teampreview actions are always valid (90 = C(6,4) * P(4,2)/2
        #      = 15 team selections * 6 lead pair orderings, but pairs are unordered)
        #    - No masking needed because any selection is legal regardless of
        #      opponent's team or game state
        #
        # 2. TURN PHASE (battle.teampreview = False):
        #    - Player chooses actions for their 2 active Pokemon each turn
        #    - 2,025 possible action combinations (45 actions per Pokemon * 45)
        #    - MANY actions are invalid based on current game state:
        #      * Pokemon may be fainted (can't give orders to dead Pokemon)
        #      * Moves may be disabled, out of PP, or locked (Choice items, etc.)
        #      * Switching may be blocked (trapped, Sky Drop, etc.)
        #      * Targets may be invalid (can't target fainted Pokemon)
        #      * Gimmicks (Mega, Z-Move, Tera) may already be used
        #    - We MUST mask invalid actions to prevent the model from:
        #      a) Wasting probability mass on impossible moves
        #      b) Selecting invalid moves that cause server errors/fallbacks
        #      c) Learning corrupted value estimates from invalid action attempts
        #
        # FORCE_SWITCH is a special case within turn phase:
        #    - When a Pokemon faints, the player must switch in a replacement
        #    - Only switch actions are valid (no moves allowed)
        #    - Handled by MDBO.FORCE_SWITCH action type
        # =====================================================================

        if battle.teampreview:
            # Teampreview: all 90 actions are valid, no mask needed
            mask = None
        else:
            # Turn actions: must validate each of 2,025 possible action combinations
            mask = self._get_action_mask(battle)

        # Create future
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Push to queue
        await self.queue.put((state, future, battle.battle_tag, battle.teampreview, mask))

        # Wait for result
        result = await future

        action_idx = result["action"]

        # Store step
        step = {
            "state": state,
            "action": action_idx,
            "log_prob": result["log_prob"],
            "value": result["value"],
            "reward": 0,  # Will be filled later
            "is_teampreview": battle.teampreview,
            "mask": mask,
        }

        if battle.battle_tag not in self.current_trajectories:
            self.current_trajectories[battle.battle_tag] = []
        self.current_trajectories[battle.battle_tag].append(step)

        # Convert to BattleOrder
        if battle.teampreview:
            try:
                mdbo = MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW)
                order = mdbo.message
            except ValueError:
                # Fallback to random team order
                # "123456" is a valid order string part, but we need "/team 123456"
                order = "/team 123456"
        else:
            try:
                mdbo = MDBO.from_int(action_idx, type=MDBO.TURN)
                order = mdbo.to_double_battle_order(battle)
            except ValueError:
                order = DefaultBattleOrder()

        return order  # type: ignore

    def _get_action_mask(self, battle: DoubleBattle) -> np.ndarray:
        """
        Build a binary mask over the 2,025 possible turn actions indicating validity.

        This is the same validation logic used in:
        - behavior_clone_player.py: BCPlayer.predict() for inference
        - battle_dataset.py: BattleDataset.__getitem__() for training labels

        Consistency is critical: if training data masks action X as invalid but
        the RL worker allows it, the model learns conflicting signals.

        Args:
            battle: Current DoubleBattle state

        Returns:
            np.ndarray of shape (2025,) with 1.0 for valid actions, 0.0 for invalid

        Performance Note:
            This iterates over all 2,025 actions and validates each one (~50-100ms).
            This is acceptable because:
            1. It only runs once per decision point (not per training step)
            2. Correctness > speed for action validity
            3. Alternative approaches (caching, precomputation) add complexity
        """
        mask = np.zeros(MDBO.action_space(), dtype=np.float32)

        # Determine action type based on game state:
        # - FORCE_SWITCH: A Pokemon fainted and must be replaced (only switches valid)
        # - TURN: Normal turn with moves and optional switches
        if any(battle.force_switch):
            action_type = MDBO.FORCE_SWITCH
        else:
            action_type = MDBO.TURN

        # Validate each possible action by attempting to convert it to a battle order
        # and checking if that order is legal in the current game state
        for i in range(MDBO.action_space()):
            try:
                # MDBO.from_int() converts integer action index to MDBO object
                # .to_double_battle_order() converts MDBO to poke-env's DoubleBattleOrder
                mdbo = MDBO.from_int(i, type=action_type)
                order = mdbo.to_double_battle_order(battle)

                # is_valid_order() checks all game rules
                if isinstance(order, DoubleBattleOrder) and is_valid_order(order, battle):
                    mask[i] = 1.0
            except (ValueError, KeyError, AttributeError):
                # Action conversion failed - this action is invalid
                # Common causes:
                # - Action refers to a Pokemon not on the field
                # - Action refers to a move the Pokemon doesn't have
                # - Action has invalid target specification
                pass

        # Safety check: ensure at least one action is valid
        # If no actions are valid, something is wrong with the game state
        if mask.sum() == 0:
            print(
                f"WARNING: No valid actions found for battle {battle.battle_tag}. "
                f"Force switch: {battle.force_switch}, Active: {battle.active_pokemon}"
            )
            # Return all-ones as fallback to avoid crashes (model will pick randomly)
            return np.ones(MDBO.action_space(), dtype=np.float32)

        return mask

    def _battle_finished_callback(self, battle: AbstractBattle):
        """
        Called automatically when a battle ends. Finalizes trajectory and pushes to training queue.

        This callback:
        1. Retrieves the trajectory (sequence of state-action-value tuples) for this battle
        2. Assigns sparse win/loss rewards (+1 for win, -1 for loss) to all steps
        3. Pushes the complete trajectory to the training queue for policy updates
        4. Cleans up memory (trajectories and LSTM hidden states)
        """
        if battle.battle_tag in self.current_trajectories:
            # Pop the trajectory we've been building during this battle
            traj = self.current_trajectories.pop(battle.battle_tag)

            # Assign sparse rewards: RNaD uses binary win/loss signal
            # All steps get the same reward (credit assignment handled by value function & GAE)
            reward = 1.0 if battle.won else -1.0
            for step in traj:
                step["reward"] = reward

            # Push to shared training queue for the learner to consume
            # This is how workers communicate completed battles to the main training loop
            if self.trajectory_queue is not None:
                self.trajectory_queue.put(traj)
            else:
                # Keep for debugging/inspection when no training queue
                self.completed_trajectories.extend(traj)

            # Clean up LSTM hidden state for this battle (no longer needed)
            # Prevents memory leak when running many battles sequentially
            if battle.battle_tag in self.hidden_states:
                del self.hidden_states[battle.battle_tag]
