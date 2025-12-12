import numpy as np

import sys; sys.path.append(".")
import utils
from feat_extractors.text_vectorizer.simple_vectorizers import CountVectorizer, TfidfVectorizer
from feat_extractors.extractors_ensemble import ensembled_extract
from classifiers.bagger.bagger_presets import get_bagger_model, train_bagger_model, predict_bagger_model


if __name__ == "__main__":
    ## Training ##

    print("Load training dataset")
    texts_train = utils.load_cmv_data("./dataset/train.jsonl")

    df_train = utils.parse_cmv_data(texts_train)

    print("Extract features")
    # vectorizer = CountVectorizer(max_feat_count=2000)  # 效果不好
    vectorizer = TfidfVectorizer(max_feat_count=2000)

    vectorizer.train(df_train['o'].values.tolist() + df_train['p'].values.tolist())

    x_train_o = ensembled_extract(df_train["o"].values.tolist(), vectorizer)
    x_train_p = ensembled_extract(df_train["p"].values.tolist(), vectorizer)
    # TODO: mini-batched extraction and/or training and evaluation

    # x_train = np.concat([x_train_o, x_train_p], axis=-1)  # 直接把楼主发言和说服者发言的向量拼起来
    x_train = x_train_p - x_train_o  # or 相减
    y_train = df_train['label'].values.astype(int)
    print(f"Feature extracted - x {x_train.shape}, y {y_train.shape}")

    print("Get and train model")
    model = get_bagger_model(xgb_ref_y_train=y_train)
    train_bagger_model(model, x_train, y_train)

    from datetime import datetime
    utils.save_model(model, f"./model/simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")

    ## Evaluation ##

    print("Load evaluation dataset")
    texts_test = utils.load_cmv_data("./dataset/val.jsonl")

    df_test = utils.parse_cmv_data(texts_test)

    print("Extract features")
    x_test_o = ensembled_extract(df_test["o"].values.tolist(), vectorizer)  # 使用训练集上的 vectorizer
    x_test_p = ensembled_extract(df_test["p"].values.tolist(), vectorizer)

    # x_test = np.concat([x_test_o, x_test_p], axis=-1)
    x_test = x_test_p - x_test_o
    y_test = df_test['label'].values.astype(int)

    print("Predict")
    predict_bagger_model(model, x_test, y_test)