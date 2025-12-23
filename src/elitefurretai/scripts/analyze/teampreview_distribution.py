#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick script to analyze teampreview action distribution in raw battle files.
"""

import os
import sys
from collections import Counter
import orjson
import math

from elitefurretai.model_utils import BattleData, BattleIterator, MDBO


def analyze_teampreview_distribution(battle_files_dir: str, num_files: int = 100):
    """
    Load N battle files and analyze the distribution of teampreview actions.

    Args:
        battle_files_dir: Directory containing raw battle JSON files
        num_files: Number of files to analyze
    """

    # Get list of files
    all_files = [
        os.path.join(battle_files_dir, f)
        for f in os.listdir(battle_files_dir)
        if f.endswith('.json')
    ][:num_files]

    print(f"Analyzing {len(all_files)} battle files...\n")

    # Track teampreview actions
    p1_actions: Counter[str] = Counter()
    p2_actions: Counter[str] = Counter()
    p1_action_ints: Counter[int] = Counter()
    p2_action_ints: Counter[int] = Counter()

    # Track team orderings in JSON
    p1_team_orders = []
    p2_team_orders = []

    for i, file_path in enumerate(all_files):
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(all_files)} files...", end='\r')

        try:
            # Load battle data
            with open(file_path, 'r') as f:
                json_data = orjson.loads(f.read())

            bd = BattleData.from_showdown_json(json_data)

            # Get p1's teampreview choice
            iter_p1 = BattleIterator(bd, perspective="p1", omniscient=True)
            if iter_p1.next_input():
                if iter_p1.last_input_type == MDBO.TEAMPREVIEW:
                    p1_order = iter_p1.last_order()
                    p1_actions[p1_order.message] += 1
                    p1_action_ints[p1_order.to_int()] += 1

            # Get p2's teampreview choice
            iter_p2 = BattleIterator(bd, perspective="p2", omniscient=True)
            if iter_p2.next_input():
                if iter_p2.last_input_type == MDBO.TEAMPREVIEW:
                    p2_order = iter_p2.last_order()
                    p2_actions[p2_order.message] += 1
                    p2_action_ints[p2_order.to_int()] += 1

            # Track team order from JSON (first 4 Pokemon indices)
            p1_team_orders.append([p.species for p in bd.p1_team[:4]])
            p2_team_orders.append([p.species for p in bd.p2_team[:4]])

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            continue

    print(f"\nProcessed {len(all_files)} files successfully!\n")

    # Print results
    print("=" * 80)
    print("TEAMPREVIEW ACTION ANALYSIS")
    print("=" * 80)

    print(f"\nP1 Teampreview Actions (Total: {sum(p1_actions.values())})")
    print(f"Unique actions: {len(p1_actions)}")
    print("Most common (top 10):")
    for action, count in p1_actions.most_common(10):
        pct = count / sum(p1_actions.values()) * 100
        print(f"  {action:30} → {count:4} times ({pct:5.2f}%)")

    print(f"\nP2 Teampreview Actions (Total: {sum(p2_actions.values())})")
    print(f"Unique actions: {len(p2_actions)}")
    print("Most common (top 10):")
    for action, count in p2_actions.most_common(10):
        pct = count / sum(p2_actions.values()) * 100
        print(f"  {action:30} → {count:4} times ({pct:5.2f}%)")

    print("\n" + "=" * 80)
    print("ACTION INTEGER DISTRIBUTION")
    print("=" * 80)

    print("\nP1 Action Integers (Top 10):")
    for action_int, count in p1_action_ints.most_common(10):
        pct = count / sum(p1_action_ints.values()) * 100
        print(f"  Action {action_int:2} → {count:4} times ({pct:5.2f}%)")

    print("\nP2 Action Integers (Top 10):")
    for action_int, count in p2_action_ints.most_common(10):
        pct = count / sum(p2_action_ints.values()) * 100
        print(f"  Action {action_int:2} → {count:4} times ({pct:5.2f}%)")

    # Check if team order in JSON matches teampreview choice
    print("\n" + "=" * 80)
    print("CHECKING FOR LEAKAGE: Do JSON team positions match teampreview choices?")
    print("=" * 80)

    # The hypothesis: if JSON stores teams in teampreview order, then
    # most battles should choose action corresponding to "1234" (first 4 Pokemon)

    # What action integer corresponds to choosing Pokemon [1,2,3,4]?
    # Based on encoder.py, this should be the ordering where first pair is (1,2) and second pair is (3,4)
    # Since v[1] > v[0] and v[3] > v[2], this is stored as "2143"
    from elitefurretai.model_utils.encoder import _TEAMPREVIEW_ORDER_TO_INT

    print(f"\nTotal possible teampreview actions: {len(_TEAMPREVIEW_ORDER_TO_INT)}")
    print("\nSample of teampreview encodings:")
    for key, val in list(_TEAMPREVIEW_ORDER_TO_INT.items())[:10]:
        print(f"  Order '{key}' → Action {val}")

    # Check what choosing "first 4 Pokemon in order" would be
    if "2143" in _TEAMPREVIEW_ORDER_TO_INT:
        first_four_action = _TEAMPREVIEW_ORDER_TO_INT["2143"]
        print("\nIf JSON teams are pre-ordered by teampreview choice:")
        print(f"  Choosing Pokemon [1,2,3,4] = Action {first_four_action}")
        print(f"  P1 frequency: {p1_action_ints[first_four_action]} / {sum(p1_action_ints.values())} = {p1_action_ints[first_four_action] / sum(p1_action_ints.values()) * 100:.1f}%")
        print(f"  P2 frequency: {p2_action_ints[first_four_action]} / {sum(p2_action_ints.values())} = {p2_action_ints[first_four_action] / sum(p2_action_ints.values()) * 100:.1f}%")

    # Calculate entropy to see how uniform the distribution is
    def entropy(counter):
        total = sum(counter.values())
        return -sum((count / total) * math.log2(count / total) for count in counter.values() if count > 0)

    max_entropy = math.log2(90)  # Uniform distribution over 90 actions
    p1_entropy = entropy(p1_action_ints)
    p2_entropy = entropy(p2_action_ints)

    print("\n" + "=" * 80)
    print("DISTRIBUTION UNIFORMITY (Entropy Analysis)")
    print("=" * 80)
    print(f"Maximum possible entropy (uniform): {max_entropy:.3f} bits")
    print(f"P1 action entropy: {p1_entropy:.3f} bits ({p1_entropy / max_entropy * 100:.1f}% of maximum)")
    print(f"P2 action entropy: {p2_entropy:.3f} bits ({p2_entropy / max_entropy * 100:.1f}% of maximum)")
    print("\nInterpretation:")
    print("  - High entropy (~6.5 bits) = Uniform distribution, no obvious pattern")
    print("  - Low entropy (<4 bits) = Clustered around few actions, potential leakage")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_files = int(sys.argv[1])
    else:
        num_files = 100

    battle_dir = "data/battles/gen9vgc2023regulationc_raw"

    if not os.path.exists(battle_dir):
        print(f"Error: Directory '{battle_dir}' not found!")
        print("Please provide the correct path to raw battle files.")
        sys.exit(1)

    analyze_teampreview_distribution(battle_dir, num_files)
