# -*- coding: utf-8 -*-
"""A wrapper class for a model
"""
from abc import ABC, abstractmethod
from typing import List


class AbstractModel(ABC):

    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def create(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError(
            """This will take in a list of battles to learn from, and process each battle, similar
            to the implementation in the original R-NaD paper. It will load battles from a list of
            filepaths using DataProcessor"""
        )

    @abstractmethod
    def save(self, filepath: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, filepath: str):
        raise NotImplementedError

    @abstractmethod
    def get_action_probabilities(self, observation: List[float]) -> List[float]:
        raise NotImplementedError
