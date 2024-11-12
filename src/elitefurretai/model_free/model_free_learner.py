# -*- coding: utf-8 -*-
"""This module isn't yet implemented. It will host R-NaD code to learn
from a replay buffer
"""
from typing import List
from weakref import WeakKeyDictionary

from elitefurretai.model_utils.battle_data import BattleData


class ModelFreeLearner:

    def __init__(self):
        self._replay_buffer: WeakKeyDictionary[str, BattleData] = WeakKeyDictionary()
        raise NotImplementedError

    def reset(self):
        self._replay_buffer = WeakKeyDictionary()

    def learn(self, keys: List[str]):
        raise NotImplementedError(
            """This will take in a list of battles to learn from, and process each battle, similar
            to the implementation in the original R-NaD paper. It will load battles from a list of
            filepaths using DataProcessor"""
        )

    def save_policy(self, filename):
        raise NotImplementedError

    def generate_battle(self):
        for tag in self._replay_buffer:
            yield self._replay_buffer[tag]
