# -*- coding: utf-8 -*-
import sys

import orjson

from elitefurretai.model_utils import (
    BattleData,
    BattleIterator,
    Embedder,
    ModelBattleOrder,
)


# This scripts reads through a BattleData file and generates training data from those files
def main(filename):
    for perspective in ["p1", "p2"]:

        bd = None
        with open(filename, "r") as f:
            bd = BattleData.from_showdown_json(orjson.loads(f.read()))

        battle = bd.to_battle(perspective)
        assert battle is not None and battle.format is not None

        embedder = Embedder(format=battle.format, simple=False)
        iter = BattleIterator(battle, bd, perspective=perspective)

        while iter.next_input() and not battle.finished:

            # Get the last input command found by the iterator
            input = iter.last_input
            if input is None:
                break

            request = iter.simulate_request()
            battle.parse_request(request)

            embedder.feature_dict_to_vector(embedder.featurize_double_battle(battle))

            # Standardize the input (remove identifiers names and convert to ints)
            order = ModelBattleOrder.from_battle_data(input, battle, bd)  # Human order
            win = int(
                bd.winner == (bd.p1 if perspective == "p1" else bd.p2)
            )  # Win prediction

            print(
                f"Got {input} -> {order.message} -> {order.to_int()} w/ {perspective} {'' if win else 'not '}winning on event #{iter.index} ({bd.logs[iter.index + 1]})"
            )
            print()

        print(f"Done with perspective {perspective}!\n")


if __name__ == "__main__":
    main(sys.argv[1])
