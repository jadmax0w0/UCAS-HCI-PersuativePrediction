import torch
from torch import Tensor
import numpy as np
from numpy.typing import NDArray

from abc import ABC
from typing import Union


class BaseTextFeatureExtractor(ABC):
    def trained(self) -> bool:
        raise NotImplementedError()
    
    def train(self, training_texts: list[str], **kwargs):
        raise NotImplementedError()
    
    def extract(self, text: Union[str, list[str]], **kwargs) -> NDArray:
        raise NotImplementedError()