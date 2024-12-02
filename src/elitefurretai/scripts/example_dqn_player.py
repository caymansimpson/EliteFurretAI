import random
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Space
from poke_env.environment import AbstractBattle, Battle, DoubleBattle
from poke_env.player import Player
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder
from poke_env.player.env_player import EnvPlayer
from poke_env.player.random_player import RandomPlayer

from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.utils.team_repo import TeamRepo


# We define our RL player
class SimpleDQNPlayer(EnvPlayer):

    # These are used by EnvPlayer
    _ACTION_SPACE = list(range(4 + 6))
    _DEFAULT_BATTLE_FORMAT = "gen9vgc2024regh"

    def __init__(self, num_battles=10000, **kwargs):
        super().__init__(**kwargs)
        self._embedder = Embedder()
        random.seed(21)

    def create_model(self):
        # Simple model where only one layer feeds into the next
        self._model = Sequential()

        # Get initializer for hidden layers
        init = tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.02)

        # Input Layer; this shape is one that just works
        self._model.add(
            Dense(
                512,
                input_shape=(1, 7814),
                activation="relu",
                use_bias=False,
                kernel_initializer=init,
                name="first_hidden",
            )
        )

        # Hidden Layers
        self._model.add(
            Flatten(name="flatten")
        )  # Flattening resolve potential issues that would arise otherwise
        self._model.add(
            Dense(
                256,
                activation="relu",
                use_bias=False,
                kernel_initializer=init,
                name="second_hidden",
            )
        )

        # Output Layer
        self._model.add(
            Dense(
                len(self._ACTION_SPACE),
                use_bias=False,
                kernel_initializer=init,
                name="final",
            )
        )
        self._model.add(
            BatchNormalization()
        )  # Increases speed: https://www.dlology.com/blog/one-simple-trick-to-train-keras-model-faster-with-batch-normalization/
        self._model.add(
            Activation("linear")
        )  # Same as passing activation in Dense Layer, but allows us to access last layer: https://stackoverflow.com/questions/40866124/difference-between-dense-and-activation-layer-in-keras

        # This is how many battles we'll remember before we start forgetting old ones
        self._memory = SequentialMemory(limit=max(num_battles, 10000), window_length=1)

        # Simple epsilon greedy policy
        # This takes the output of our NeuralNet and converts it to a value
        # Softmax is another probabilistic option: https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py#L120
        self._policy = LinearAnnealedPolicy(
            MaxBoltzmannQPolicy(),
            attr="eps",
            value_max=1.0,
            value_min=0.05,
            value_test=0,
            nb_steps=num_battles,
        )

        # Defining our DQN
        self._dqn = DQNAgent(
            model=self._model,
            nb_actions=len(action_space),
            policy=self._policy,
            memory=self._memory,
            nb_steps_warmup=max(
                1000, int(num_battles / 10)
            ),  # The number of battles we go through before we start training: https://hub.packtpub.com/build-reinforcement-learning-agent-in-keras-tutorial/
            gamma=0.8,  # This is the discount factor for the Value we learn - we care a lot about future rewards
            target_model_update=0.01,  # This controls how much/when our model updates: https://github.com/keras-rl/keras-rl/issues/55
            delta_clip=1,  # Helps define Huber loss - cips values to be -1 < x < 1. https://srome.github.io/A-Tour-Of-Gotchas-When-Implementing-Deep-Q-Networks-With-Keras-And-OpenAi-Gym/
            enable_double_dqn=True,
        )

        self._dqn.compile(Adam(lr=0.01), metrics=["mae"])

    # Takes the output of our policy (which chooses from a `_ACTION_SPACE`-dimensional array), and converts it into a battle order
    def _action_to_single_move(self, action: int, index: int, battle):

        if action < 24:
            # If either there is no mon or we're forced to switch, there's nothing to do
            if not battle.active_pokemon[index] or battle.force_switch[index]:
                return None
            dynamax, remaining = action % 2 == 1, int(action / 2)
            if battle.active_pokemon[index] and int(remaining / 3) < len(
                battle.active_pokemon[index].moves
            ):
                move, initial_target = (
                    list(battle.active_pokemon[index].moves.values())[int(remaining / 3)],
                    remaining % 3,
                )

                # If there's no target needed, we create the action as we normally would. It doesn't matter what our AI returned as target since there's only one possible target
                if move.deduced_target not in [
                    "adjacentAlly",
                    "adjacentAllyOrSelf",
                    "any",
                    "normal",
                ]:
                    return BattleOrder(
                        order=move, actor=battle.active_pokemon[index], dynamax=dynamax
                    )

                # If we are targeting a single mon, there are three cases: your other mon, the opponents mon or their other mon.
                # 2 corresponds to your mon and 0/1 correspond to the opponents mons (index in opponent_active_mon)
                # For the self-taret case, we ensure there's another mon on our side to hit (otherwise we leave action1 as None)
                elif initial_target == 2:
                    if battle.active_pokemon[1] is not None:
                        return BattleOrder(
                            order=move,
                            move_target=battle.active_pokemon_to_showdown_target(
                                1 - index, opp=False
                            ),
                            actor=battle.active_pokemon[index],
                            dynamax=dynamax,
                        )

                # In the last case (if initial_target is 0 or 1), we target the opponent, and we do it regardless of what slot was
                # chosen if there's only 1 mon left. In the following cases, we handle whether there are two mons left or one mon left
                elif len(battle.opponent_active_pokemon) == 2 and all(
                    battle.opponent_active_pokemon
                ):
                    return BattleOrder(
                        order=move,
                        move_target=battle.active_pokemon_to_showdown_target(
                            initial_target, opp=True
                        ),
                        actor=battle.active_pokemon[index],
                        dynamax=dynamax,
                    )
                elif len(battle.opponent_active_pokemon) < 2 and any(
                    battle.opponent_active_pokemon
                ):
                    initial_target = (
                        1 if battle.opponent_active_pokemon[0] is not None else 0
                    )
                    return BattleOrder(
                        order=move,
                        move_target=battle.active_pokemon_to_showdown_target(
                            initial_target, opp=True
                        ),
                        actor=battle.active_pokemon[index],
                        dynamax=dynamax,
                    )

        elif 25 - action < len(battle.available_switches[index]):
            return BattleOrder(
                order=battle.available_switches[index][25 - action],
                actor=battle.active_pokemon[index],
            )

        return None

    # Takes the output of our policy (which chooses from a 676-dimensional array), and converts it into a battle order
    def _action_to_move(self, action: int, battle: Battle) -> str:  # pyre-ignore
        """Converts actions to move orders. There are 676 actions - and they can be thought of as a 26 x 26 matrix (first mon's possibilities
        and second mon's possibilities). This is not quite true because you cant choose the same mon twice to switch to, but we handle that when
        determining the legality of the move choices later; If the proposed action is illegal, a random legal move is performed.
        The conversion is done as follows:

        :param action: The action to convert.
        :type action: int
        :param battle: The battle in which to act.
        :type battle: Battle
        :return: the order to send to the server.
        :rtype: str
        """
        row, col = action % 26, int(action / 26)
        first_order = (
            self._action_to_single_move(row, 0, battle)
            if battle.active_pokemon[0]
            else None
        )
        second_order = (
            self._action_to_single_move(col, 1, battle)
            if battle.active_pokemon[1]
            else None
        )

        double_order = DoubleBattleOrder(
            first_order=first_order, second_order=second_order
        )
        if DoubleBattleOrder.is_valid(battle, double_order):
            return double_order
        else:
            return DefaultBattleOrder()

    def describe_embedding(self) -> Space[ObsType]:
        """
        Returns the description of the embedding. It must return a Space specifying
        low bounds and high bounds.

        :return: The description of the embedding.
        :rtype: Space
        """
        return Box(
            np.array([-100] * self._embedder.embedding_size, dtype=np.float32),
            np.array([100] * self._embedder.embedding_size, dtype=np.float32),
            dtype=np.float32,
        )

    def calc_reward(
        self, last_battle: AbstractBattle, current_battle: AbstractBattle
    ) -> float:
        """
        Returns the reward for the current battle state. The battle state in the previous
        turn is given as well and can be used for comparisons.

        :param last_battle: The battle state in the previous turn.
        :type last_battle: AbstractBattle
        :param current_battle: The current battle state.
        :type current_battle: AbstractBattle

        :return: The reward for current_battle.
        :rtype: float
        """
        if current_battle.won:
            return 1
        elif current_battle.lost:
            return -1
        return 0

    def embed_battle(self, battle: AbstractBattle) -> ObsType:
        emb = self._embedder.featurize_double_battle(battle)  # type: ignore
        return np.float32(self.feature_dict_to_vector(emb))

    def compute_reward(self, battle: Union[Battle, DoubleBattle]) -> float:
        """A helper function to compute rewards. We only give rewards for winning
        :param battle: The battle for which to compute rewards.
        :type battle: Union[Battle, DoubleBattle]
        :return: the reward
        :rtype: float
        """

        # Victory condition
        if battle.won:
            reward = 1
        elif battle.lost:
            reward = -1
        else:
            reward = 0

        self._reward_buffer[battle] = reward

        return reward

    # Because of env_player implementation, it requires an initial parameter passed, in this case, it's the object itself (player == self)
    def _training_helper(self, player, num_steps=10000):
        self._dqn.fit(self, nb_steps=num_steps)
        self.complete_current_battle()

    def train(self, opponent: Player, num_steps: int) -> None:
        self.play_against(
            env_algorithm=self._training_helper,
            opponent=opponent,
            env_algorithm_kwargs={"num_steps": num_steps},
        )

    def save_model(self, filename=None) -> None:
        if filename is not None:
            self._dqn.save_weights("models/" + filename, overwrite=True)
        else:
            self._dqn.save_weights(
                "models/model_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                overwrite=True,
            )

    def load_model(self, filename: str) -> None:
        self._dqn.load_weights("models/" + filename)

    def evaluate_model(self, num_battles: int, v=True) -> float:
        self.reset_battles()
        self._dqn.test(nb_episodes=num_battles, visualize=False, verbose=False)
        if v:
            print(
                "DQN Evaluation: %d wins out of %d battles"
                % (self.n_won_battles, num_battles)
            )
        return self.n_won_battles * 1.0 / num_battles

    def choose_move(self, battle: AbstractBattle) -> str:
        if battle not in self._observations or battle not in self._actions:
            self._init_battle(battle)
        self._observations[battle].put(self.embed_battle(battle))
        action = self._actions[battle].get()
        order = self._action_to_move(action, battle)

        return order.message

    # Deterministic since this is a toy example
    def teampreview(self, battle):
        return "/team 1234"

    @property
    def model(self):
        """
        Return our Keras-trained model
        """
        return self._model

    @property
    def memory(self) -> List:
        """
        Return the memory for our DQN
        """
        return self._memory

    @property
    def policy(self) -> List:
        """
        Return our policy for our DQN
        """
        return self._policy

    @property
    def dqn(self) -> List:
        """
        Return our DQN object
        """
        return self._dqn


# TODO: train this model and evaluate it
# also add embedder and embeddding past 10 turns using reuniclus
async def main():
    tr = TeamRepo()
    efai = SimpleDQNPlayer(
        AccountConfiguration("elitefurretai", None),
        battle_format="gen9vgc2024regg",
        team=tr.get("gen9vgc2024regh", "dqn"),
    )
    p2 = RandomPlayer()
    p3 = MaxDamagePlayer()  # TODO: implement or pull

    # First test the environment to ensure the class is consistent
    # with the OpenAI API
    opponent = RandomPlayer(battle_format="gen8randombattle")
    test_env = SimpleRLPlayer(
        battle_format="gen8randombattle", start_challenging=True, opponent=opponent
    )
    check_env(test_env)
    test_env.close()

    # Create one environment for training and one for evaluation
    opponent = RandomPlayer(battle_format="gen8randombattle")
    train_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )
    opponent = RandomPlayer(battle_format="gen8randombattle")
    eval_env = SimpleRLPlayer(
        battle_format="gen8randombattle", opponent=opponent, start_challenging=True
    )

    # Compute dimensions
    n_action = train_env.action_space.n
    input_shape = (1,) + train_env.observation_space.shape

    # Create model
    model = Sequential()
    model.add(Dense(128, activation="elu", input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(n_action, activation="linear"))

    # Defining the DQN
    memory = SequentialMemory(limit=10000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.05,
        value_test=0.0,
        nb_steps=10000,
    )

    dqn = DQNAgent(
        model=model,
        nb_actions=n_action,
        policy=policy,
        memory=memory,
        nb_steps_warmup=1000,
        gamma=0.5,
        target_model_update=1,
        delta_clip=0.01,
        enable_double_dqn=True,
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # Training the model
    dqn.fit(train_env, nb_steps=10000)
    train_env.close()

    # Evaluating the model
    print("Results against random player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    second_opponent = MaxBasePowerPlayer(battle_format="gen8randombattle")
    eval_env.reset_env(restart=True, opponent=second_opponent)
    print("Results against max base power player:")
    dqn.test(eval_env, nb_episodes=100, verbose=False, visualize=False)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.reset_env(restart=False)

    # Evaluate the player with included util method
    n_challenges = 250
    placement_battles = 40
    eval_task = background_evaluate_player(eval_env.agent, n_challenges, placement_battles)
    dqn.test(eval_env, nb_episodes=n_challenges, verbose=False, visualize=False)
    print("Evaluation with included method:", eval_task.result())
    eval_env.reset_env(restart=False)

    # Cross evaluate the player with included util method
    n_challenges = 50
    players = [
        eval_env.agent,
        RandomPlayer(battle_format="gen8randombattle"),
        MaxBasePowerPlayer(battle_format="gen8randombattle"),
        SimpleHeuristicsPlayer(battle_format="gen8randombattle"),
    ]
    cross_eval_task = background_cross_evaluate(players, n_challenges)
    dqn.test(
        eval_env,
        nb_episodes=n_challenges * (len(players) - 1),
        verbose=False,
        visualize=False,
    )
    cross_evaluation = cross_eval_task.result()
    table = [["-"] + [p.username for p in players]]
    for p_1, results in cross_evaluation.items():
        table.append([p_1] + [cross_evaluation[p_1][p_2] for p_2 in results])
    print("Cross evaluation of DQN with baselines:")
    print(tabulate(table))
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
