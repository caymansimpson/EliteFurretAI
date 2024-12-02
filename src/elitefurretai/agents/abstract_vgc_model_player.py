from abc import ABC, abstractmethod
from typing import List

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder


# We define our RL player
class AbstractVGCModelPlayer(Player, ABC):

    def __init__(self, **kwargs):
        self._last_message_error = False
        super().__init__(**kwargs)

    @abstractmethod
    def create_model(self, **kwargs):
        pass

    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        pass

    @abstractmethod
    def action_to_move(self, action: int) -> BattleOrder:
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass

    @abstractmethod
    def embed_battle_state(self, battle: AbstractBattle) -> List[float]:
        pass

    @abstractmethod
    def get_observation(self, battle: AbstractBattle) -> List[float]:
        pass

    async def send_message(self, message: str, room: str):
        await self.ps_client.send_message(room, message)

    # Wrote some basic unnecessary code to dictate whether the last message was an error
    # TODO: Need to be able to understand when I get an error, wait for a next request
    # and then after I process the request, then send a next move. This is detailed here:
    # https://github.com/smogon/pokemon-showdown/blob/cb9c45c4ffdd189b1ee3ef5e4095f7c1bde17b34/sim/SIM-PROTOCOL.md
    async def handle_battle_message(self, split_messages: List[List[str]]):
        if split_messages[0][0] == "error" and split_messages[0][1] in [
            "[Unavailable choice]",
            "[Invalid choice]",
        ]:
            self._last_message_error = True
        else:
            self._last_message_error = False
        await super()._handle_battle_message(split_messages)

    @property
    def last_message_error(self) -> bool:
        return self._last_message_error

    @property
    @abstractmethod
    def action_space(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> int:
        pass

    @abstractmethod
    def calculate_reward(self, battle: AbstractBattle) -> float:
        pass

    @abstractmethod
    def get_info(self, battle: AbstractBattle) -> dict:
        pass
