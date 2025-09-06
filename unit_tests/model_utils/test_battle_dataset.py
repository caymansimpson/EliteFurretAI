import os
import tempfile

import orjson
import pytest

from elitefurretai.model_utils import (
    MDBO,
    BattleData,
    BattleDataset,
    BattleIterator,
    Embedder,
)
from elitefurretai.utils.battle_order_validator import is_valid_order

# List of all vgc_json_anon fixtures in conftest.py
VGC_JSON_FIXTURES = [
    "vgc_json_anon",
    "vgc_json_anon2",
    "vgc_json_anon3",
    "vgc_json_anon4",
    "vgc_json_anon5",
    "vgc_json_anon6",
    "vgc_json_anon7",
    "vgc_json_anon8",
    "vgc_json_anon9",
    "vgc_json_anon10",
    "vgc_json_anon11",
    "vgc_json_anon12",
    "vgc_json_anon13",
    "vgc_json_anon14",
    "vgc_json_anon15",
    "vgc_json_anon16",
    "vgc_json_anon17",
    "vgc_json_anon18",
    "vgc_json_anon19",
    "vgc_json_anon20",
    "vgc_json_anon21",
    "vgc_json_anon22",
    "vgc_json_anon23",
]


@pytest.mark.parametrize("fixture_name", VGC_JSON_FIXTURES)
def test_battle_dataset_actions_and_wins(request, fixture_name):
    battle_json = request.getfixturevalue(fixture_name)

    # Write the JSON to a temporary file, as BattleDataset expects filenames
    with tempfile.NamedTemporaryFile("wb", delete=False) as f:
        f.write(orjson.dumps(battle_json))
        temp_path = f.name

    try:
        embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set="full", omniscient=True
        )
        dataset = BattleDataset([temp_path], embedder=embedder, steps_per_battle=17)

        for idx in range(2):
            metrics = dataset[idx]
            states = metrics["states"]
            actions = metrics["actions"]
            action_masks = metrics["action_masks"]
            wins = metrics["wins"]
            masks = metrics["masks"]
            kos = metrics["kos"]
            switches = metrics["switches"]

            assert states is not None
            assert kos is not None
            assert switches is not None

            bd = BattleData.from_showdown_json(battle_json)
            perspective = "p" + str((idx % 2) + 1)
            iter = BattleIterator(bd, perspective=perspective, omniscient=True)

            for i in range(actions.shape[0]):
                if masks[i] == 0 or iter.last_input is None:
                    continue

                action_idx = int(actions[i].item())
                assert (
                    action_masks[i, action_idx] == 1
                ), f"Step {i}: Action {action_idx} not valid in action_masks for {fixture_name}, perspective {perspective}"

                input_type = iter.last_input_type
                order = MDBO.from_int(action_idx, input_type).to_double_battle_order(iter.battle)  # type: ignore
                assert is_valid_order(order, iter.battle), (  # type: ignore
                    f"Step {i}: Action {action_idx} not valid according to is_valid_order for {fixture_name}, perspective {perspective}"
                )

                expected_win = int(bd.winner == iter.battle.player_username)
                assert (
                    int(wins[i].item()) == expected_win
                ), f"Step {i}: Win label {wins[i].item()} incorrect for {fixture_name}, perspective {perspective}"
                iter.next_input()
    finally:
        os.remove(temp_path)
