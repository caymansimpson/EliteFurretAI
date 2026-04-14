from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from poke_env.battle import DoubleBattle

from elitefurretai.rl.players import MaxDamagePlayer


def test_score_available_actions_filters_moves_not_in_request():
    player = MaxDamagePlayer.__new__(MaxDamagePlayer)
    player.debug = False
    player.switch_threshold = 100.0
    player.create_order = lambda move, move_target=None: SimpleNamespace(order=move, move_target=move_target)
    player._get_best_move_damage = lambda battle, switch_mon: (0.0, None)

    battle = MagicMock(spec=DoubleBattle)
    battle.last_request = {
        "active": [
            {
                "moves": [
                    {"id": "meteorbeam", "target": "normal", "pp": 8, "disabled": False}
                ]
            },
            {"moves": []},
        ]
    }

    stale_move = SimpleNamespace(id="terastarstorm")
    legal_move = SimpleNamespace(id="meteorbeam")
    battle.available_moves = [[stale_move, legal_move], []]

    active_mon = MagicMock()
    active_mon.species = "Terapagos"
    active_mon.identifier.return_value = "p1: Terapagos"
    battle.active_pokemon = [active_mon, None]

    opponent = MagicMock()
    opponent.species = "Volcarona"
    opponent.identifier.return_value = "p2: Volcarona"
    battle.opponent_active_pokemon = [opponent, None]
    battle.available_switches = [[], []]
    battle.player_role = "p1"
    battle.opponent_role = "p2"

    with patch("elitefurretai.rl.players.calculate_damage", return_value=(10, 10)):
        candidates = player._score_available_actions(battle, 0, set())

    assert candidates
    assert all(candidate[0].order.id == "meteorbeam" for candidate in candidates)
