import sys; sys.path.append(".")
from feat_extractors.text_vectorizer.simple_vectorizers import CountVectorizer
from feat_extractors.bert_extractor import BertTextFeatureExtractor
from classifiers.bagger.bagger_presets import get_bagger_model
from pipeline import train_model, eval_model


if __name__ == "__main__":
    model = get_bagger_model(enable_decision_tree=False, enable_random_forest=False, enable_xgb=False)

    count_vec = CountVectorizer(max_feat_count=1000)
    bert_ext = BertTextFeatureExtractor("E:/Storage/AIModels/bert-base-multilingual-cased", minibatch_size=128)

    train_model(model, "dataset/train.jsonl", count_vec, bert_ext, o_p_integrate_method='subtract', model_save_path='model/bag_LsS.pkl')

    eval_model('model/bag_LsS.pkl', "dataset/val.jsonl", count_vec, bert_ext, o_p_integrate_method='subtract')