import sys; sys.path.append(".")
from feat_extractors.cialdini_extractor import CialdiniFeatureExtractor
from feat_extractors.dim10_extractor import Dim10FeatureExtractor
from feat_extractors.bert_extractor import BertTextFeatureExtractor
from classifiers.bagger.bagger_presets import get_bagger_model
from pipeline import train_model, eval_model


if __name__ == "__main__":
    model = get_bagger_model(enable_xgb=False)

    raise NotImplementedError("看到这行报错说明你还没改掉 bert 检查点的路径, 在 main.py 里面改一下")
    bert_ext = BertTextFeatureExtractor("/path/to/bert/ckpt", minibatch_size=128)
    cial_ext = CialdiniFeatureExtractor()  # TODO: cialdini extractor WIP
    dm10_ext = Dim10FeatureExtractor()

    train_model(
        model=model,
        data_path="dataset/train.jsonl",
        cialdini_extractor="dataset/cialdini_scores_train.json",
        dim10_extractor=dm10_ext,
        bert_extractor=bert_ext,
        model_save_path="model/test.pkl"
    )

    eval_model(
        model_path=model,
        data_path="dataset/val.jsonl",
        cialdini_extractor="dataset/cialdini_scores_val.json",
        dim10_extractor=dm10_ext,
        bert_extractor=bert_ext,
    )