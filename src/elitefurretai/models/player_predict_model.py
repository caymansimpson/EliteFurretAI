# -*- coding: utf-8 -*-
"""A wrapper class for a model
"""
from abc import abstractmethod
from typing import List

from elitefurretai.models.abstract_model import AbstractModel


class PlayerPredictModel(AbstractModel):

    def __init__(self):
        pass

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

    @staticmethod
    def load(filepath: str):
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation: List[float]) -> List[float]:
        raise NotImplementedError
