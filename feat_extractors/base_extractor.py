import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray

from abc import ABC
from typing import Union


class BaseTextFeatureExtractor(ABC):
    def train(self, training_texts: list[str]):
        raise NotImplementedError()
    
    def extract(self, text: Union[str, list[str]]) -> NDArray:
        raise NotImplementedError()