import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Literal

import sys; sys.path.append(".")
from classifiers.base_classifier import BaseClassifier
from feat_extractors.base_extractor import BaseTextFeatureExtractor
from feat_extractors.extractors_ensemble import ensembled_extract


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


def train_model(
        model: BaseClassifier,
        data_path: str,
        *feat_extractors: BaseTextFeatureExtractor,
        o_p_integrate_method: Literal['concat', 'subtract'] = 'concat',
):
    """
    Args:
        o_p_integrate_method: 怎么样处理向量化后的楼主 & 说服者发言. `"concat"` = 直接拼接, `"subtract"` = 相减
    """
    print("Load training dataset")
    texts_train = load_cmv_data("./dataset/train.jsonl")
    df_train = parse_cmv_data(texts_train)

    for ext in tqdm(feat_extractors, desc="Train feature extractors", total=len(feat_extractors)):
        ext.train(df_train['o'].values.tolist() + df_train['p'].values.tolist())

    x_train_o = ensembled_extract(df_train["o"].values.tolist(), *feat_extractors)
    x_train_p = ensembled_extract(df_train["p"].values.tolist(), *feat_extractors)
    # TODO: mini-batched extraction and/or training and evaluation

    if o_p_integrate_method == "concat":
        x_train = np.concat([x_train_o, x_train_p], axis=-1)  # 直接把楼主发言和说服者发言的向量拼起来
    elif o_p_integrate_method == "subtract":
        x_train = x_train_p - x_train_o  # or 相减
    else:
        raise ValueError(f"Unknown original and persuasive integrate method: {o_p_integrate_method}")

    y_train = df_train['label'].values.astype(int)
    print(f"Feature extracted - x {x_train.shape}, y {y_train.shape}")


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
