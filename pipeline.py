import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Literal, Union
from sklearn.metrics import classification_report

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


def parse_cmv_data(rawdata: list[dict], max_char_count: Optional[int] = None, max_data_count: Optional[int] = None):
    """
    Args:
        max_char_count: debug 用，设置每个字符串的最大字符数
        max_data_count: debug 用，设置最多读取多少轮对话
    Returns:
        DataFrame: columns: `['o', 'p', 'label']`, 分别对应楼主文本、说服者文本、说服是 (1) 否 (0) 成功
    """
    def parse_line(dataline: dict):
        o = str(dataline['o_title']) + "\n\n" + str(dataline['o_text'])
        pp = "\n\n".join(dataline['p_positive_texts'])
        pn = "\n\n".join(dataline['p_negative_texts'])
        if max_char_count is not None:
            o = o[:max_char_count]
            pp = pp[:max_char_count]
            pn = pn[:max_char_count]
        return [[o, pp, 1], [o, pn, 0]]
    
    samples = []
    rawdata_trunc = rawdata if max_data_count is None else rawdata[:max_data_count]
    for dataline in rawdata_trunc:
        sample = parse_line(dataline)
        sample = np.array(sample, dtype=object)
        samples.append(sample)
    
    samples = np.concat(samples, axis=0)
    return pd.DataFrame(samples, columns=['o', 'p', 'label'])


def train_model(
        model: BaseClassifier,
        data_path: Optional[str],
        *feat_extractors: BaseTextFeatureExtractor,
        o_p_integrate_method: Literal['concat', 'subtract'] = 'concat',
        model_save_path: Optional[str] = None,
):
    """
    Args:
        o_p_integrate_method: 怎么样处理向量化后的楼主 & 说服者发言. `"concat"` = 直接拼接, `"subtract"` = 相减
    Returns:
        tuple: (trained model, trained feat. extractors)
    """
    print("Load training dataset")
    data_path = "./dataset/train.jsonl" if data_path is None else data_path
    texts_train = load_cmv_data(data_path)
    df_train = parse_cmv_data(texts_train, max_char_count=200, max_data_count=500)  # TODO: 上量 & 解决 bert 输入串长度问题

    for ext in tqdm(feat_extractors, desc="Train feature extractors", total=len(feat_extractors)):
        ext.train(df_train['o'].values.tolist() + df_train['p'].values.tolist())

    x_train_o = ensembled_extract(df_train["o"].values.tolist(), *feat_extractors)
    x_train_p = ensembled_extract(df_train["p"].values.tolist(), *feat_extractors)

    if o_p_integrate_method == "concat":
        x_train = np.concat([x_train_o, x_train_p], axis=-1)  # 直接把楼主发言和说服者发言的向量拼起来
    elif o_p_integrate_method == "subtract":
        x_train = x_train_p - x_train_o  # or 相减
    else:
        raise ValueError(f"Unknown original and persuasive integrate method: {o_p_integrate_method}")

    y_train = df_train['label'].values.astype(int)
    print(f"Feature extracted - x {x_train.shape}, y {y_train.shape}")

    print("Train model")
    model.train(x_train, y_train)

    if model_save_path is not None:
        save_model(model, model_save_path)
        print(f"Model saved at {model_save_path}")
    
    print("Train complete")
    return model, feat_extractors


def eval_model(
        model_path: Union[BaseClassifier, str],
        data_path: Optional[str],
        *feat_extractors: BaseTextFeatureExtractor,
        o_p_integrate_method: Literal['concat', 'subtract'] = 'concat',
):
    """
    Args:
        o_p_integrate_method: 怎么样处理向量化后的楼主 & 说服者发言. `"concat"` = 直接拼接, `"subtract"` = 相减
    Note:
        确保 `feat_extractors` 是 `train_model()` 中训练过的那些 & 确保 `o_p_integrate_method` 与 `train_model()` 一致
    """
    if isinstance(model_path, str):
        print("Load model checkpoint")
        model = load_model(model_path, return_only_model=True)
    else:
        model = model_path

    print("Load evaluation dataset")
    data_path = "./dataset/val.jsonl" if data_path is None else data_path
    texts_test = load_cmv_data(data_path)
    df_test = parse_cmv_data(texts_test, max_char_count=200, max_data_count=500)

    print("Extract features")
    assert all(ext.trained() for ext in feat_extractors), "Not all feat. extractors are trained"
    x_test_o = ensembled_extract(df_test["o"].values.tolist(), *feat_extractors)
    x_test_p = ensembled_extract(df_test["p"].values.tolist(), *feat_extractors)

    y_test = df_test['label'].values.astype(int)
    if o_p_integrate_method == "concat":
        x_test = np.concat([x_test_o, x_test_p], axis=-1)
    elif o_p_integrate_method == "subtract":
        x_test = x_test_p - x_test_o
    else:
        raise ValueError(f"Unknown original and persuasive integrate method: {o_p_integrate_method}")

    print("Model predict")
    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))


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
