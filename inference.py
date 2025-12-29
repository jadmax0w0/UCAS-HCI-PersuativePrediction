import numpy as np

import sys; sys.path.append(".")
from feat_extractors.cialdini_extractor import CialdiniFeatureExtractor
from feat_extractors.dim10_extractor import Dim10FeatureExtractor
from feat_extractors.bert_extractor import BertTextFeatureExtractor
from classifiers.bagger.bagger_presets import get_bagger_model
from classifiers.bagger.voting_bagger import VotingBagger
from pipeline import train_model, eval_model, load_model, get_custom_text_features


O = "I hate human."

P = "No you don't."


if __name__ == "__main__":
    model = get_bagger_model(enable_xgb=False)

    bert_ext = BertTextFeatureExtractor("/Users/youxseem/Documents/AIModels.localized/bert-base-multilingual-cased", minibatch_size=128)
    cial_ext = CialdiniFeatureExtractor()  # TODO: cialdini extractor WIP
    dm10_ext = Dim10FeatureExtractor()

    model = load_model("model/test2.pkl")
    assert isinstance(model, VotingBagger)
    bert_ext.train()
    cial_ext.train()

    # o = input("Opinion:\n")
    # p = input("Persuasive:\n")
    # cial = input("Cialdini (split with comma)\n[Reciprocity, Consistency, Social_Proof, Authority, Scarcity, Liking]:\n")
    # cial = np.array([int(v) for v in cial.split(",")]).reshape(1, -1)
    cial = np.array([[0,0,0,0,0,0]])
    o = O
    p = P
    o_feat = get_custom_text_features(
        text=o,
        bert_extractor=bert_ext,
    )
    p_feat = get_custom_text_features(
        text=p,
        cialdini_extractor=cial,
        dim10_extractor=dm10_ext,
        bert_extractor=bert_ext,
    )
    feat = np.concat([p_feat, o_feat], axis=-1)
    print(f"{feat.shape=}")

    y_pred = model.predict(feat)
    print(f"{y_pred=}")