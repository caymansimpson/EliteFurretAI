# -*- coding: utf-8 -*-
"""This script trains a supervised model to predict who will win a battle using a simple model in sklearn. If you
have your own replays, you can build a model on your own dataset.
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

from elitefurretai.model_utils import BattleDataset


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
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


def get_hms(start):
    total_time = time.time() - start
    hours = int(total_time / (60 * 60))
    mins = int((total_time % (60 * 60)) / 60)
    secs = int(total_time % 60)
    return hours, mins, secs


# Arguments: link to json object with BattleData filepaths and number of battles to train/evalute on
def main():
    total_battles = int(sys.argv[2])
    test_split = 0.1
    batch_size = 32
    eval_size = 100

    # For tracking progress through training
    start = time.time()
    benchmark = start

    # Takes 18m with 5000 battles and 100 evaluation
    print(f"\nStarting! Preparing dataset of {total_battles}...")
    files, eval_files = [], [],
    with open(sys.argv[1], "rb") as f:
        files = sorted(orjson.loads(f.read()), key=lambda x: random.random())
        eval_files = files[int(total_battles / 2):int(total_battles / 2) + eval_size]
        files = files[:int(total_battles / 2)]

    # Generate data
    dataset = BattleDataset(files)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=min(os.cpu_count() or 1, 4)
    )

    hours, mins, secs = get_hms(benchmark)
    print(f"Finished preparing dataset in {hours}h {mins}m {secs}s!! Now loading data...")
    benchmark = time.time()

    X_list, y_list, i = [], [], 0
    for states, _, _, wins, masks in data_loader:

        # Reshape to 2D (batch_size * steps, features)
        batch_size, steps, features = states.shape
        X_flat = states.view(-1, features).numpy()

        # Flatten masks and wins
        mask_flat = masks.view(-1).bool().numpy()
        y_flat = wins.view(-1).long().numpy()

        # Apply masks
        X_list.append(X_flat[mask_flat])
        y_list.append(y_flat[mask_flat])

        hours, minutes, seconds = get_hms(benchmark)
        processed = f"Processed {i * batch_size} battles ({round(i * batch_size * 100.0 / total_battles, 2)}%) in {hours}h {minutes}m {seconds}s"
        print("\r" + processed + "     ", end="")
        i += 1

    # Combine all batches
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=21
    )

    hours, mins, secs = get_hms(benchmark)
    print(
        f"\nFinished loading data in {hours}h {mins}m {secs}s with {len(y)} steps! Now training..."
    )
    benchmark = time.time()

    # Train
    model = HistGradientBoostingClassifier(
        learning_rate=0.01,
        early_stopping=True,
        max_features=.01,
        n_iter_no_change=10,
        l2_regularization=1.0,
        max_depth=5,
        max_iter=2000,
        min_samples_leaf=1000,
        validation_fraction=0.2,
    )
    model.fit(X_train, y_train)

    hours, mins, secs = get_hms(benchmark)
    print(f"Finished training in {hours}h {mins}m {secs}s! Results:")
    benchmark = time.time()
    evaluate(model.predict(X_train), y_train)

    print("Now going to evaluate model on test set!")
    evaluate(model.predict(X_test), y_test)

    hours, mins, secs = get_hms(benchmark)
    print(f"Finished evaluating on test set in {hours}h {mins}m {secs}s! Results:")
    benchmark = time.time()

    # Evaluate data
    print(f"Now going to evaluate model on eval set of {eval_size} entirely new battles!")
    eval_data_loader = DataLoader(BattleDataset(eval_files), batch_size=batch_size, num_workers=min(os.cpu_count() or 1, 4))

    X_list, y_list, i = [], [], 0
    for states, _, _, wins, masks in eval_data_loader:

        # Reshape to 2D (batch_size * steps, features)
        batch_size, steps, features = states.shape
        X_flat = states.view(-1, features).numpy()

        # Flatten masks and wins
        mask_flat = masks.view(-1).bool().numpy()
        y_flat = wins.view(-1).long().numpy()

        # Apply masks
        X_list.append(X_flat[mask_flat])
        y_list.append(y_flat[mask_flat])

        hours, minutes, seconds = get_hms(benchmark)
        processed = f"Processed {i * batch_size} battles ({round(i * batch_size * 100.0 / len(eval_files), 2)}%) in {hours}h {minutes}m {seconds}s"
        print("\r" + processed + "     ", end="")
        i += 1

    # Combine all batches
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    hours, mins, secs = get_hms(benchmark)
    print(f"Finished loading evaluation data in {hours}h {mins}m {secs}s! Evaluation Results:\n")
    benchmark = time.time()

    evaluate(model.predict(X), y)
    hours, mins, secs = get_hms(benchmark)
    print(f"Finished evaluating in {hours}h {mins}m {secs}s!")
    hours, mins, secs = get_hms(start)
    print(f"Done in {hours}h {mins}m {secs}s!")


if __name__ == "__main__":
    main()
