import asyncio

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.player.random_player import RandomPlayer

from elitefurretai.utils.team_repo import TeamRepo
from elitefurretai.agents.simple_vgc_dqn_player import SimpleVGCDQNPlayer
from elitefurretai.environment.vgc_environment import VGCEnvironment


# TODO: train this model and evaluate it
# also add embedder and embeddding past 10 turns using reuniclus
async def main():
    tr = TeamRepo()

    # TODO: create arbitrary policy
    efai_policy = DQNPolicy(
        **kwargs
    )

    # TODO: create arbitrary model
    efai_model = DQN(
        efai_policy,
        **kwargs
    )

    # TODO: pull in player arguments manually
    # Create Agents
    efai = SimpleVGCDQNPlayer(
        AccountConfiguration("elitefurretai", None),
        battle_format="gen9vgc2024regh",
        simple=True,
        probabilistic=False,
        efai_model = efai_model
        team=tr.get("gen9vgc2024regh", "dqn"),
    )

    opponent = RandomPlayer(battle_format="gen9vgc2024regh")

    # Create one environment for training and one for evaluation
    train_env = VGCEnvironment(battle_format="gen9vgc2024regh", agents=[efai, opponent])

    # Train the model
    # TODO: fix
    assert efai.model is not None
    efai.model.learn(train_env, total_timesteps==10000)
    train_env.close()

    # Evaluating the model against RandomPlayer
    eval_env = VGCEnvironment(battle_format="gen9vgc2024regh", agents=[efai, opponent])
    print("Results against random player:")
    eval_env.evaluate(nb_episodes=100)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.close()

    # Evaluating the model against MaxDamagePlayer
    eval_env = VGCEnvironment(battle_format="gen9vgc2024regh", agents=[efai, opponent])
    second_opponent = MaxDamagePlayer(battle_format="gen9vgc2024regh")    
    print("Results against max base power player:")
    eval_env.evaluate(nb_episodes=100)
    print(
        f"DQN Evaluation: {eval_env.n_won_battles} victories out of {eval_env.n_finished_battles} episodes"
    )
    eval_env.close()


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
