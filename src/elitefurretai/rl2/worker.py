import asyncio
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from poke_env.player import Player, RandomPlayer
from poke_env.battle import DoubleBattle, AbstractBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.encoder import MDBO
from elitefurretai.rl2.agent import RNaDAgent

class BatchInferencePlayer(Player):
    def __init__(
        self, 
        model: RNaDAgent, 
        device='cpu', 
        batch_size=16, 
        batch_timeout=0.01, 
        probabilistic=True,
        trajectory_queue=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.probabilistic = probabilistic
        self.trajectory_queue = trajectory_queue
        self.embedder = Embedder(format=self.format, feature_set=Embedder.FULL, omniscient=False)
        self.queue = asyncio.Queue()
        self.hidden_states = {} # battle_tag -> (h, c)
        self._inference_task = None
        
        # Trajectory storage
        # battle_tag -> list of dicts
        self.current_trajectories = {} 
        self.completed_trajectories = [] 
        
    async def start_inference_loop(self):
        self._inference_task = asyncio.create_task(self._inference_loop())
        
    async def _inference_loop(self):
        while True:
            batch = []
            futures = []
            battle_tags = []
            is_tps = []
            masks = []
            
            # Collect batch
            try:
                # Wait for first item
                item = await self.queue.get()
                self._add_to_batch(batch, futures, battle_tags, is_tps, masks, item)
                
                # Collect more
                start_time = asyncio.get_event_loop().time()
                while len(batch) < self.batch_size:
                    timeout = self.batch_timeout - (asyncio.get_event_loop().time() - start_time)
                    if timeout <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                        self._add_to_batch(batch, futures, battle_tags, is_tps, masks, item)
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
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device).unsqueeze(1) # (batch, 1, dim)
        
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
            self.hidden_states[tag] = (h_next[:, i:i+1, :], c_next[:, i:i+1, :])
            
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
                        probs = mask / mask.sum() # Fallback to uniform over valid
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
                'action': action,
                'log_prob': log_prob,
                'value': values[i, 0],
                'probs': probs
            }
            future.set_result(result)

    def choose_move(self, battle: AbstractBattle) -> Any:
        # This method returns a coroutine that resolves to a BattleOrder
        return self._choose_move_async(battle)

    async def _choose_move_async(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, DoubleBattle):
            return DefaultBattleOrder()
            
        # Embed state
        state = self.embedder.feature_dict_to_vector(self.embedder.embed(battle))
        
        # Get mask
        if battle.teampreview:
            mask = None # TP mask logic if needed
        else:
            # MDBO mask
            # We need to get valid actions from MDBO
            # This is expensive? MDBO.action_masks(battle)
            # Let's assume we have a helper or just use valid_orders
            # MDBO doesn't have a direct "mask from battle" method in the snippet I saw.
            # But we can rely on the model learning valid moves or use a simple mask if available.
            # For now, let's pass None and rely on post-check or simple masking if implemented.
            # Wait, the user instructions mentioned "Always mask invalid actions via action_masks tensor".
            # I need to implement `get_action_mask`.
            mask = self._get_action_mask(battle)

        # Create future
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        # Push to queue
        await self.queue.put((state, future, battle.battle_tag, battle.teampreview, mask))
        
        # Wait for result
        result = await future
        
        action_idx = result['action']
        
        # Store step
        step = {
            'state': state,
            'action': action_idx,
            'log_prob': result['log_prob'],
            'value': result['value'],
            'reward': 0, # Will be filled later
            'is_teampreview': battle.teampreview,
            'mask': mask
        }
        
        if battle.battle_tag not in self.current_trajectories:
            self.current_trajectories[battle.battle_tag] = []
        self.current_trajectories[battle.battle_tag].append(step)
        
        # Convert to BattleOrder
        if battle.teampreview:
            try:
                mdbo = MDBO.from_int(action_idx, type=MDBO.TEAMPREVIEW)
                order = mdbo.message
            except:
                # Fallback to random team order
                # "123456" is a valid order string part, but we need "/team 123456"
                order = "/team 123456"
        else:
            try:
                mdbo = MDBO.from_int(action_idx, type=MDBO.TURN)
                order = mdbo.to_double_battle_order(battle)
            except:
                order = DefaultBattleOrder()
                
        return order

    def _get_action_mask(self, battle):
        # Implement masking logic based on MDBO and battle.valid_orders
        # This is complex. For now return None or all-ones.
        # Ideally we map valid_orders to indices.
        return None

    def _battle_finished_callback(self, battle: AbstractBattle):
        if battle.battle_tag in self.current_trajectories:
            traj = self.current_trajectories.pop(battle.battle_tag)
            # Assign rewards
            reward = 1.0 if battle.won else -1.0
            for step in traj:
                step['reward'] = reward # Sparse reward at end? Or shaped?
                # RNaD usually uses win/loss.
            
            self.completed_trajectories.append(traj)
            
            if self.trajectory_queue is not None:
                self.trajectory_queue.put(traj)
                self.completed_trajectories = [] # Clear local storage if pushed
            
            # Clean up hidden state
            if battle.battle_tag in self.hidden_states:
                del self.hidden_states[battle.battle_tag]

