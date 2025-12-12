import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Optional, List
import sys; sys.path.append(".")
from classifiers.base_classifier import BaseClassifier


def load_cmv_data(path: str):
    import json

    if ".jsonl" in path.lower():
        with open(path, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines]
    
    elif ".json" in path.lower():
        with open(path, mode='r', encoding='utf-8') as f:
            return json.loads(f.read())
    
    else:
        raise NotImplementedError(f"Unsupported data file: {path}")


def parse_cmv_data(rawdata: list[dict]):
    """
    Returns:
        DataFrame: columns: `['o', 'p', 'label']`, 分别对应楼主文本、说服者文本、说服是 (1) 否 (0) 成功
    """
    def parse_line(dataline: dict):
        o = str(dataline['o_title']) + "\n\n" + str(dataline['o_text'])
        pp = "\n\n".join(dataline['p_positive_texts'])
        pn = "\n\n".join(dataline['p_negative_texts'])
        return [[o, pp, 1], [o, pn, 0]]
    
    samples = []
    for dataline in rawdata:
        sample = parse_line(dataline)
        sample = np.array(sample, dtype=object)
        samples.append(sample)
    
    samples = np.concat(samples, axis=0)
    return pd.DataFrame(samples, columns=['o', 'p', 'label'])


def save_model(model: BaseClassifier, path: str):
    if path is None:
        return None
    
    import pickle
    import os
    from datetime import datetime

    fdir, _ = os.path.split(path)
    if fdir != "":
        os.makedirs(fdir, exist_ok=True)
    
    with open(path, mode="wb") as f:
        pickle.dump({"model": model, "train_time": datetime.now()}, f)


def load_model(path: str, return_only_model = True):
    if path is None:
        return None
    
    import pickle
    with open(path, mode="rb") as f:
        model = pickle.load(f)
    return model['model'] if return_only_model else model
