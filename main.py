import sys; sys.path.append(".")
from feat_extractors.cialdini_extractor import CialdiniFeatureExtractor
from feat_extractors.bert_extractor import BertTextFeatureExtractor
from classifiers.bagger.bagger_presets import get_bagger_model
from pipeline import train_model, eval_model


if __name__ == "__main__":
    model = get_bagger_model(enable_decision_tree=False, enable_random_forest=False, enable_xgb=False)

    bert_ext = BertTextFeatureExtractor("E:/Storage/AIModels/bert-base-multilingual-cased", minibatch_size=128)
    cial_ext = CialdiniFeatureExtractor()  # TODO: cialdini extractor WIP

    train_model(
        model=model,
        data_path="dataset/train.jsonl",
        cialdini_extractor="dataset/cialdini_scores_train.json",
        dim10_extractor=None,
        bert_extractor=bert_ext,
        model_save_path="model/test.pkl"
    )

    eval_model(
        model_path=model,
        data_path="dataset/val.json",
        cialdini_extractor=cial_ext,  # TODO: 等着换成标注好的 cialdini scores for val set
        dim10_extractor=None,
        bert_extractor=bert_ext,
    )