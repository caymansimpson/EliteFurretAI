# -*- coding: utf-8 -*-
"""This script analyzes VGC logs
"""
import sys
import time
import random

from poke_env.environment.move import Move

from elitefurretai.model_utils.training_generator import file_generator


def main(dir):
    effects = {}
    abilities = {}
    items = {}
    turns = {}
    choices = {}
    all_effects = set()
    all_abilities = set()
    all_items = set()

    # Initialize data structures
    for rating in range(950, 1900, 50):
        effects[rating] = {}
        abilities[rating] = {}
        items[rating] = {}
        turns[rating] = {}
        choices[rating] = {}

    # For tracking
    start_time = time.time()
    count = 0

    # Go through files
    for bd, filename in file_generator(dir, slice=(0, 1), sort=lambda x: random.random()):

        # For each player, record ability and items and choices
        for perspective in ["p1", "p2"]:
            battle = bd.to_battle(perspective)
            rating = int((bd.p1_rating if perspective == "p1" else bd.p2_rating) / 50) * 50
            if rating < 1000 or rating > 1850:
                continue

            # Go through teams for mon information
            for mon in battle.team.values():
                abilities[rating][mon.ability] = abilities[rating].get(mon.ability, 0) + 1
                items[rating][mon.item] = items[rating].get(mon.item, 0) + 1
                all_abilities.add(mon.ability)
                all_items.add(mon.item)

            # Go through inputs for choice information
            for input_log in bd.input_logs:
                if not input_log.startswith(">" + perspective) or "team" in input_log or ("pass" in input_log and "switch" in input_log):
                    continue

                for input in input_log.split(", "):
                    if input.startswith(">"):
                        input = input[4:]
                    input = input.strip()

                    if "default" in input:
                        pass
                    elif "switch" in input:
                        choices[rating]["switch"] = choices[rating].get("switch", 0) + 1
                    elif "move" in input:
                        move = Move(input.split(" ")[1], gen=battle.gen)
                        if move.base_power > 0:
                            choices[rating]["damage"] = choices[rating].get("damage", 0) + 1
                        else:
                            choices[rating]["status"] = choices[rating].get("status", 0) + 1

        # Look at battle statistics
        avg_rating = int((bd.p1_rating + bd.p2_rating) / 2 / 50) * 50
        turns[avg_rating][bd.turns] = turns[avg_rating].get(bd.turns, 0) + 1

        # Look at effects by iterating through logs
        for raw_log in bd.logs:
            log = raw_log.split("|")
            if len(log) < 3:
                continue

            if log[1] in ["-start", "-activate", "-singleturn", "-singlemove"] and len(log) >= 3:
                rating = int((bd.p1_rating if log[2].startswith("p1") else bd.p2_rating) / 50) * 50
                effects[rating][log[3]] = effects[avg_rating].get(log[3], 0) + 1
                all_effects.add(log[3])

        count += 1
        if count % 10 == 0:
            hours = int(time.time() - start_time) // 3600
            minutes = int(time.time() - start_time) // 60
            seconds = int(time.time() - start_time) % 60
            print(f"\rProcessed {count} batles in {hours}h {minutes}m {seconds}s", end="")

    # Add 0s for missing data
    for rating in effects:
        for effect in all_effects:
            effects[rating][effect] = effects[rating].get(effect, 0)

        for ability in all_abilities:
            abilities[rating][ability] = abilities[rating].get(ability, 0)

        for item in all_items:
            items[rating][item] = items[rating].get(item, 0)

    # Save results
    with open("/Users/cayman/Desktop/effects.csv", "w") as f:
        f.write("rating,effect,count\n")
        for rating in effects:
            for effect, count in effects[rating].items():
                f.write(f"{rating},{effect},{count}\n")

    with open("/Users/cayman/Desktop/abilities.csv", "w") as f:
        f.write("rating,ability,count\n")
        for rating in abilities:
            for ability, count in abilities[rating].items():
                f.write(f"{rating},{ability},{count}\n")

    with open("/Users/cayman/Desktop/items.csv", "w") as f:
        f.write("rating,item,count\n")
        for rating in items:
            for item, count in items[rating].items():
                f.write(f"{rating},{item},{count}\n")

    with open("/Users/cayman/Desktop/turns.csv", "w") as f:
        f.write("rating,turn,count\n")
        for rating in turns:
            for turn, count in turns[rating].items():
                f.write(f"{rating},{turn},{count}\n")

    with open("/Users/cayman/Desktop/choices.csv", "w") as f:
        f.write("rating,choice,count\n")
        for rating in choices:
            for choice, count in choices[rating].items():
                f.write(f"{rating},{choice},{count}\n")

    print()
    print(f"Done reading {count} battles in {round(time.time() - start_time, 2)} seconds")


if __name__ == "__main__":
    main(sys.argv[1])
