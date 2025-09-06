# -*- coding: utf-8 -*-
"""
This script trains a supervised model to predict who will win a battle using a simple model in sklearn.
If you have your own replays, you can build a model on your own dataset.
"""

import os
import random
import sys
import time

import numpy as np
import orjson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from elitefurretai.model_utils import BattleDataset, format_time


def evaluate(y_true, y_pred):
    """
    Print a variety of classification metrics for model evaluation.
    """
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    print("Confusion Matrix")
    print(cm)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    print("ROC AUC:", roc_auc)
    print()


# Arguments: link to json object with BattleData filepaths and number of battles to train/evalute on
# Example usage: python examples/sklearn_classifier.py /path/to/battle_files.json 10000
def main():
    # Parse command line arguments
    total_battles = int(sys.argv[2])  # Number of battles to use
    test_split = 0.1  # Fraction of data for test set
    batch_size = 32  # Batch size for DataLoader
    eval_size = 100  # Number of battles for final evaluation

    # For tracking progress through training
    start = time.time()
    benchmark = start

    # Load file paths from JSON file and shuffle
    print(f"\nStarting! Preparing dataset of {total_battles}...")
    files, eval_files = [], []
    with open(sys.argv[1], "rb") as f:
        # Shuffle files randomly
        files = sorted(orjson.loads(f.read()), key=lambda x: random.random())
        # Select a slice for evaluation and the rest for training
        eval_files = files[int(total_battles / 2) : int(total_battles / 2) + eval_size]
        files = files[: int(total_battles / 2)]

    # Generate data using BattleDataset and DataLoader for batching
    dataset = BattleDataset(files)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=min(os.cpu_count() or 1, 4)
    )

    print(f"Finished preparing dataset in {format_time(time.time() - benchmark)}! Now loading data...")
    benchmark = time.time()

    # Lists to accumulate features and labels from all batches
    X_list, y_list, i = [], [], 0
    for metrics in data_loader:
        states = metrics["states"]
        wins = metrics["wins"]
        masks = metrics["masks"]

        # states: (batch_size, steps, features)
        # Reshape to 2D (batch_size * steps, features)
        batch_size, steps, features = states.shape
        X_flat = states.view(-1, features).numpy()

        # Flatten masks and wins to 1D
        mask_flat = masks.view(-1).bool().numpy()
        y_flat = wins.view(-1).long().numpy()

        # Apply masks to select only valid samples
        X_list.append(X_flat[mask_flat])
        y_list.append(y_flat[mask_flat])

        # Print progress
        processed = f"Processed {i * batch_size} battles ({round(i * batch_size * 100.0 / total_battles, 2)}%) in {format_time(time.time() - benchmark)}"
        print("\r" + processed + "     ", end="")
        i += 1

    # Combine all batches into single arrays for sklearn
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Split into train/test sets for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=21
    )

    print(
        f"\nFinished loading data in {format_time(time.time() - benchmark)} with {len(y)} steps! Now training..."
    )
    benchmark = time.time()

    # Train a HistGradientBoostingClassifier (fast, robust tree-based model)
    model = HistGradientBoostingClassifier(
        learning_rate=0.01,
        early_stopping=True,
        max_features=0.01,  # Use a small fraction of features per split for speed
        n_iter_no_change=10,
        l2_regularization=1.0,
        max_depth=5,
        max_iter=2000,
        min_samples_leaf=1000,  # Large leaf size for regularization
        validation_fraction=0.2,
    )
    model.fit(X_train, y_train)

    print(f"Finished training in {format_time(time.time() - benchmark)}! Results:")
    benchmark = time.time()
    # Evaluate on training set
    evaluate(model.predict(X_train), y_train)

    print("Now going to evaluate model on test set!")
    # Evaluate on test set
    evaluate(model.predict(X_test), y_test)

    print(f"Finished evaluating on test set in {format_time(time.time() - benchmark)}! Results:")
    benchmark = time.time()

    # Evaluate on a separate evaluation set of new battles
    print(f"Now going to evaluate model on eval set of {eval_size} entirely new battles!")
    eval_data_loader = DataLoader(
        BattleDataset(eval_files),
        batch_size=batch_size,
        num_workers=min(os.cpu_count() or 1, 4),
    )

    X_list, y_list, i = [], [], 0
    for metrics in eval_data_loader:
        states = metrics["states"]
        wins = metrics["wins"]
        masks = metrics["masks"]

        # Reshape to 2D (batch_size * steps, features)
        batch_size, steps, features = states.shape
        X_flat = states.view(-1, features).numpy()

        # Flatten masks and wins
        mask_flat = masks.view(-1).bool().numpy()
        y_flat = wins.view(-1).long().numpy()

        # Apply masks
        X_list.append(X_flat[mask_flat])
        y_list.append(y_flat[mask_flat])

        # Print progress
        processed = f"Processed {i * batch_size} battles ({round(i * batch_size * 100.0 / len(eval_files), 2)}%) in {format_time(time.time() - benchmark)}"
        print("\r" + processed + "     ", end="")
        i += 1

    # Combine all batches for evaluation
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(
        f"Finished loading evaluation data in {format_time(time.time() - benchmark)}! Evaluation Results:\n"
    )
    benchmark = time.time()

    # Evaluate on the evaluation set
    evaluate(model.predict(X), y)
    print(f"Finished evaluating in {format_time(time.time() - benchmark)}!")
    print(f"Done in {format_time(time.time() - start)}!")


if __name__ == "__main__":
    main()
