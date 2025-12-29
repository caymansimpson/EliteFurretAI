#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose teampreview data leakage by checking if Pokemon positions in the
state encoding correlate with the teampreview action label.

If positions in state → action mapping is deterministic, we have leakage.
"""

import argparse
from collections import defaultdict

import torch

from elitefurretai.etl import Embedder
from elitefurretai.etl.battle_dataloader import OptimizedBattleDataLoader


def diagnose_leakage(val_data_path: str, max_samples: int = 10000):
    """
    Check if teampreview actions are deterministically encoded in state features.
    """

    print("=" * 80)
    print("TEAMPREVIEW LEAKAGE DIAGNOSTIC")
    print("=" * 80)
    print(f"\nValidation data: {val_data_path}")
    print(f"Max samples: {max_samples}\n")

    # Initialize embedder
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
    )

    print(f"Embedder size: {embedder.embedding_size}")
    print(f"Feature names (first 50): {embedder.feature_names[:50]}")
    print(f"Feature names (last 50): {embedder.feature_names[-50:]}\n")

    # Load validation data
    print("Loading validation data...")
    val_loader = OptimizedBattleDataLoader(
        val_data_path,
        embedder=embedder,
        batch_size=512,
        num_workers=4,
        prefetch_factor=2,
        files_per_worker=1,
    )

    # Get teampreview index
    feature_names = {name: i for i, name in enumerate(embedder.feature_names)}
    teampreview_idx = feature_names["teampreview"]

    print(f"Teampreview feature index: {teampreview_idx}\n")

    # Track state → action mappings
    state_to_actions: defaultdict[tuple, list[int]] = defaultdict(list)
    action_counts: defaultdict[int, int] = defaultdict(int)

    total_samples: int = 0

    print("Collecting teampreview samples...")
    for batch_idx, batch in enumerate(val_loader):
        if total_samples >= max_samples:
            break

        states = batch["states"]
        actions = batch["actions"]

        batch_size, seq_len, _ = states.shape

        # Find teampreview steps
        for i in range(batch_size):
            for j in range(seq_len):
                if states[i, j, teampreview_idx] > 0.5:
                    # This is a teampreview step
                    state = states[i, j]
                    action = actions[i, j].item()

                    # Create a hash of the FULL state to check if anything differs
                    state_hash = tuple(state.numpy().round(3))

                    state_to_actions[state_hash].append(action)
                    action_counts[action] += 1

                    # Also save a few full states for detailed inspection
                    if total_samples < 5:
                        torch.save(state, f"debug_teampreview_state_{total_samples}.pt")
                        print(f"Saved state {total_samples}, action={action}")

                    total_samples += 1

                    if total_samples >= max_samples:
                        break
            if total_samples >= max_samples:
                break

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {total_samples} teampreview samples...")

    print(f"\nCollected {total_samples} teampreview samples")
    print(f"Unique states: {len(state_to_actions)}")
    print(f"Unique actions: {len(action_counts)}")

    # Check for determinism
    print("\n" + "=" * 80)
    print("CHECKING FOR DETERMINISTIC STATE → ACTION MAPPING")
    print("=" * 80)

    deterministic_states = 0
    non_deterministic_states = 0

    for state_hash, actions in state_to_actions.items():
        unique_actions = set(actions)
        if len(unique_actions) == 1:
            deterministic_states += 1
        else:
            non_deterministic_states += 1

    total_states = len(state_to_actions)
    det_pct = (deterministic_states / total_states * 100) if total_states > 0 else 0

    print(
        f"\nDeterministic states (1 action per state): {deterministic_states} ({det_pct:.1f}%)"
    )
    print(f"Non-deterministic states (multiple actions): {non_deterministic_states}")

    if det_pct > 95:
        print("\n⚠️  WARNING: >95% of states map to exactly ONE action!")
        print("This indicates SEVERE DATA LEAKAGE - the action is encoded in the state!")
    elif det_pct > 80:
        print("\n⚠️  WARNING: >80% of states map to exactly ONE action!")
        print("This indicates probable data leakage.")
    else:
        print("\n✓ States map to multiple actions - no obvious deterministic leakage")

    # Show examples of non-deterministic states
    if non_deterministic_states > 0:
        print("\n" + "=" * 80)
        print("EXAMPLES OF NON-DETERMINISTIC STATES (same state → different actions)")
        print("=" * 80)

        count = 0
        for state_hash, actions in state_to_actions.items():
            unique_actions = set(actions)
            if len(unique_actions) > 1:
                print(f"\nState hash: {str(state_hash)[:80]}...")
                print(f"  Actions: {dict((a, actions.count(a)) for a in unique_actions)}")
                count += 1
                if count >= 5:
                    break

    # Action distribution
    print("\n" + "=" * 80)
    print("ACTION DISTRIBUTION")
    print("=" * 80)

    print(f"\nTotal actions: {len(action_counts)}")
    print("Most common actions (top 10):")
    for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        pct = count / total_samples * 100
        print(f"  Action {action:2}: {count:5} times ({pct:5.2f}%)")

    # Check for trivial baselines
    most_common_action, most_common_count = max(action_counts.items(), key=lambda x: x[1])
    baseline_acc = most_common_count / total_samples * 100

    print(
        f"\nBaseline accuracy (always predict most common action {most_common_action}): {baseline_acc:.2f}%"
    )

    if baseline_acc > 50:
        print("⚠️  WARNING: Baseline accuracy >50% suggests data imbalance")

    # Check if states are actually unique or if there are duplicates
    print("\n" + "=" * 80)
    print("STATE UNIQUENESS CHECK")
    print("=" * 80)

    duplicate_states = sum(1 for actions in state_to_actions.values() if len(actions) > 1)
    unique_states = len(state_to_actions)

    print(f"\nTotal samples: {total_samples}")
    print(f"Unique state hashes: {unique_states}")
    print(f"States seen multiple times: {duplicate_states}")
    print(f"Average samples per state: {total_samples / unique_states:.2f}")

    if total_samples / unique_states < 1.1:
        print("\n⚠️  WARNING: Almost every state is unique!")
        print("This suggests states might include battle-specific information")
        print(
            "that makes each sample unique (like exact HP values, specific Pokemon IDs, etc.)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose teampreview data leakage")
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/battles/regc_traj/val",
        help="Path to validation data directory",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10000, help="Maximum samples to process"
    )

    args = parser.parse_args()

    # Set multiprocessing strategy for WSL
    torch.multiprocessing.set_sharing_strategy("file_system")

    diagnose_leakage(
        val_data_path=args.val_data,
        max_samples=args.max_samples,
    )
