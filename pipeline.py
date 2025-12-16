import numpy as np
from numpy.typing import NDArray
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Literal, Union, Any
from sklearn.metrics import classification_report

import sys; sys.path.append(".")
from classifiers.base_classifier import BaseClassifier
import feat_extractors.cialdini_extractor as Cialdini
from feat_extractors.cialdini_extractor import CialdiniFeatureExtractor
from feat_extractors.dim10_extractor import Dim10FeatureExtractor
from feat_extractors.bert_extractor import BertTextFeatureExtractor


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
        cialdini_extractor: Union[CialdiniFeatureExtractor, str],
        dim10_extractor: Dim10FeatureExtractor,
        bert_extractor: BertTextFeatureExtractor,
        model_save_path: Optional[str] = None,
        extractors_save_dir: Optional[bool] = None,
):
    """
    Args:
        cialdini_extractor: 给定路径，则从路径读取标注好的特征；给定 extractor, 则调用 extractor
        extractors_save_dir: 是否把 extractor 也作为 ckpt 保存，不过暂时应该用不到
    Returns:
        tuple: (trained model, trained feat. extractors)
    """
    ## Load training data
    print("Load training dataset")
    data_path = "./dataset/train.jsonl" if data_path is None else data_path
    texts_train = load_cmv_data(data_path)
    df_train = parse_cmv_data(texts_train, max_char_count=100, max_data_count=300)  # TODO: 上量

    ## Get text features
    print("Prepare features")
    # Cialdini feats
    if isinstance(cialdini_extractor, CialdiniFeatureExtractor):
        p_feat_cialdini = cialdini_extractor.extract(df_train["p"].values.tolist())
    else:
        p_feat_p, p_feat_n = Cialdini.load_from_anno_file(cialdini_extractor, partition='train')
        assert p_feat_p.shape == p_feat_n.shape, f"{p_feat_p.shape=}, {p_feat_n.shape=}"
        p_feat_cialdini = np.concat([p_feat_p, p_feat_n], axis=-1).reshape(-1, p_feat_p.shape[1])
        p_feat_cialdini = p_feat_cialdini[:len(df_train)]
    
    # 10 dim feats
    if not dim10_extractor.trained():
        dim10_extractor.train()
    p_feat_dim10 = dim10_extractor.extract(df_train["p"].values.tolist())
    
    # Bert feats
    if not bert_extractor.trained():
        bert_extractor.train()
    p_feat_bert = bert_extractor.extract(df_train["p"].values.tolist())
    o_feat_bert = bert_extractor.extract(df_train["o"].values.tolist())

    # Concat feats
    x_train = np.concat([p_feat_cialdini, p_feat_dim10, p_feat_bert, o_feat_bert], axis=-1)

    y_train = df_train['label'].values.astype(int)
    print(f"Feature extracted - x {x_train.shape}, y {y_train.shape}")

    ## Train classification model
    print("Train model")
    model.train(x_train, y_train)

    if model_save_path is not None:
        save_model(model, model_save_path)
        print(f"Model saved at {model_save_path}")
    
    print("Train complete")
    return model


def eval_model(
        model_path: Union[BaseClassifier, str],
        data_path: Optional[str],
        cialdini_extractor: CialdiniFeatureExtractor,
        dim10_extractor: Dim10FeatureExtractor,
        bert_extractor: BertTextFeatureExtractor,
):
    """
    Args:
        o_p_integrate_method: 怎么样处理向量化后的楼主 & 说服者发言. `"concat"` = 直接拼接, `"subtract"` = 相减
    Returns:
        预测结果 NDArray, shaped `[N]`, `N` 为验证集样本数
    Note:
        确保 `feat_extractors` 是 `train_model()` 中训练过的那些 & 确保 `o_p_integrate_method` 与 `train_model()` 一致
    """
    if isinstance(model_path, str):
        print("Load model checkpoint")
        model = load_model(model_path, return_only_model=True)
    else:
        model = model_path

    ## Load eval data
    print("Load evaluation dataset")
    data_path = "./dataset/val.jsonl" if data_path is None else data_path
    texts_test = load_cmv_data(data_path)
    df_test = parse_cmv_data(texts_test, max_char_count=100, max_data_count=300)

    # Get features
    print("Prepare features")
    # Cialdini feats
    if isinstance(cialdini_extractor, CialdiniFeatureExtractor):
        p_feat_cialdini = cialdini_extractor.extract(df_test["p"].values.tolist())
    else:
        p_feat_p, p_feat_n = Cialdini.load_from_anno_file(cialdini_extractor, partition='val')
        assert p_feat_p.shape == p_feat_n.shape, f"{p_feat_p.shape=}, {p_feat_n.shape=}"
        p_feat_cialdini = np.concat([p_feat_p, p_feat_n], axis=-1).reshape(-1, p_feat_p.shape[1])
        p_feat_cialdini = p_feat_cialdini[:len(df_test)]
    
    # 10 dim feats
    p_feat_dim10 = dim10_extractor.extract(df_test["p"].values.tolist())
    
    # Bert feats
    p_feat_bert = bert_extractor.extract(df_test["p"].values.tolist())
    o_feat_bert = bert_extractor.extract(df_test["o"].values.tolist())

    # Concat feats
    x_test = np.concat([p_feat_cialdini, p_feat_dim10, p_feat_bert, o_feat_bert], axis=-1)

    y_test = df_test['label'].values.astype(int)
    print(f"Feature extracted - x {x_test.shape}, y {y_test.shape}")

    ## Prediction
    print("Model predict")
    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred))
    return y_pred


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
