import numpy as np
from typing import Union

import sys; sys.path.append(".")
from feat_extractors.base_extractor import BaseTextFeatureExtractor


def ensembled_extract(text: Union[str, list[str]], *extractors: BaseTextFeatureExtractor):
    feats = [ext.extract(text) for ext in extractors]
    return np.concat(feats, axis=-1)
