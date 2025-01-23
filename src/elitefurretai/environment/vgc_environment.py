import asyncio
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box, Discrete, Space
from pettingzoo.utils.env import ParallelEnv
from poke_env.environment import DoubleBattle
from poke_env.player import BattleOrder

from elitefurretai.agents.abstract_vgc_model_player import AbstractVGCModelPlayer
from elitefurretai.utils.inference_utils import observation_to_str


# We define our RL Environment
# Actions are battle orders, agentid are player usernames, observations are list of floats (embedded battles)
class VGCEnvironment(ParallelEnv):

    def __init__(
        self,
        battle_format="gen9vgc2024regh",
        agents: List[AbstractVGCModelPlayer] = [],
        seed: int = 21,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Make sure there are only two agents and they are unique
        assert len(agents) == 2
        assert len(agents) == len(set(map(lambda x: x.username, agents)))

        # Record necessary information
        self.battle_format = battle_format
        self._current_battle: Optional[DoubleBattle] = None
        self._players: Dict[str, AbstractVGCModelPlayer] = {
            agent.username: agent for agent in agents
        }
        np.random.seed(seed)

        # Record ParallelEnv information
        self.agents: List[str] = list(map(lambda x: x.username, agents))
        self.observation_spaces: Dict[str, Space] = {
            agent_id: Box(low=-100, high=100, shape=(agent.observation_space,))
            for agent_id, agent in self._players.items()
        }
        self.action_spaces: Dict[str, Space] = {
            agent_id: Discrete(agent.action_space)
            for agent_id, agent in self._players.items()
        }

    def reset(
        self, seed=None, options=None
    ) -> Tuple[Dict[str, List[float]], Dict[str, Dict]]:

        # Boilerplate recommended by https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        super().reset(seed=seed if seed is not None else int(time.time()))

        # First have to stop the battle
        if self.current_battle is not None and not self.current_battle.finished:
            first_player = list(self._players.values())[0]
            asyncio.run(
                first_player.send_message("/forfeit", self.current_battle.battle_tag)
            )

            count = 5
            wait_time = 0.1
            while self.current_battle and not self.current_battle.finished and count > 0:
                time.sleep(wait_time)
                count -= 1

            if count == 0:
                raise RuntimeError(
                    f"Cannot forfeit battle after waiting {count * wait_time} seconds"
                )

        # TODO: start battle and set to current battle
        raise NotImplementedError

    def step(self, actions: Dict[str, BattleOrder]):
        """
        # reference in openaigymenv.step definition
        """

        if not self.current_battle:
            obs, info = self.reset()
            return (
                obs,
                {agent: 0.0 for agent in self.agents},
                {agent: False for agent in self.agents},
                {agent: False for agent in self.agents},
                info,
            )

        if self.current_battle.finished:
            raise RuntimeError(
                "Battle is already finished, so we cannot step. Should have called reset"
            )

        # Step the battle state
        # TODO: make sure that this is prone to errors; figure out how to listen for errors
        asyncio.gather(
            *(
                self._players[agent].send_message(
                    action.message, self.current_battle.battle_tag
                )
                for agent, action in actions.items()
            )
        )

        # Calculate all the necessary information to return in step for each agent
        obs = {
            agent: player.get_observation(self.current_battle)
            for agent, player in self._players.items()
        }
        rewards = {
            agent: player.calculate_reward(self.current_battle)
            for agent, player in self._players.items()
        }
        info = {
            agent: player.get_info(self.current_battle)
            for agent, player in self._players.items()
        }

        # Determine whether the battle was terminated or truncated early
        # TODO: need to handle forfeits as well
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        if self.current_battle.finished:
            size = self.current_battle.team_size
            remaining_mons = size - len(
                [mon for mon in self.current_battle.team.values() if mon.fainted]
            )
            remaining_opponent_mons = size - len(
                [mon for mon in self.current_battle.opponent_team.values() if mon.fainted]
            )
            if (remaining_mons == 0) != (remaining_opponent_mons == 0):
                terminated = {agent: True for agent in self.agents}
            else:
                truncated = {agent: True for agent in self.agents}

        return obs, rewards, terminated, truncated, info

    def close(self):
        # have agents send forfeit orders for current_battle if exists and not finished
        if self.current_battle and not self.current_battle.finished:
            first_player = list(self._players.values())[0]
            asyncio.run(
                first_player.send_message("/forfeit", self.current_battle.battle_tag)
            )

            count = 5
            wait_time = 0.1
            while self.current_battle and not self.current_battle.finished and count > 0:
                time.sleep(wait_time)
                count -= 1

            if count == 0:
                raise RuntimeError(
                    f"Cannot forfeit battle after waiting {count * wait_time} seconds"
                )

        for player in self._players.values():
            player.reset_battles()

    def render(self, mode: str = "human"):
        b = self._current_battle
        if b is None:
            print("No Current Battle to render")
        else:
            obs = b.current_observation
            if obs.events == 0:
                obs = b.observations[max(b.observations.keys())]

            print(observation_to_str(obs))

    # Unnecessary for parallelenv, but I include it for convenience
    def get_agent(self, agent_id: str) -> AbstractVGCModelPlayer:
        if agent_id in self._players:
            return self._players[agent_id]
        raise ValueError(f"Unknown agent {agent_id}")

    @property
    def current_battle(self) -> Optional[DoubleBattle]:
        return self._current_battle

    def observation_space(self, agent: str) -> Space:
        if agent in self.observation_spaces:
            return self.observation_spaces[agent]
        raise ValueError(f"Unknown agent {agent}")

    def action_space(self, agent: str) -> Space:
        if agent in self.action_spaces:
            return self.action_spaces[agent]
        raise ValueError(f"Unknown agent {agent}")
