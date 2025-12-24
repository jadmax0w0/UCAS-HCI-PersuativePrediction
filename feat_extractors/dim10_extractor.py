## 💡 Adopted from feat_extractors.model_dim10.predict_10dims.py ##

import os
import torch
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional, Union
from nltk.tokenize import TweetTokenizer

import sys; sys.path.append(".")
from feat_extractors.base_extractor import BaseTextFeatureExtractor
from feat_extractors.model_dim10.preprocess_data import preprocessText, padBatch
from feat_extractors.model_dim10.preprocess_embedding import glove4gensim
from feat_extractors.model_dim10.features import ExtractWordEmbeddings
from feat_extractors.model_dim10.models.lstm import LSTMClassifier
from utils import log


DIMS_DEFAULT = [
    "social_support","conflict","trust","fun","similarity",
    "identity","respect","romance","knowledge","power"
]


class Dim10FeatureExtractor(BaseTextFeatureExtractor):
    def __init__(
            self,
            dim_names: list[str] = DIMS_DEFAULT,
            embedding_dir: str = "feat_extractors/model_dim10/weights/embeddings",
            lstm_model_dir: str = "feat_extractors/model_dim10/weights/LSTM",
            hidden_dim: int = 300,
            minibatch_size: int = 60,
            device: Optional[str] = None,
    ):
        super().__init__()

        self.dim_names = dim_names

        self.embedding_dir = embedding_dir
        self.lstm_model_dir = lstm_model_dir

        self.D = hidden_dim
        self.B = minibatch_size

        self.tokenizer = None
        self.embedding = None
        self.models = None

        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def trained(self):
        return self.tokenizer is not None and self.embedding is not None and self.models is not None
    
    def train(
            self,
            embedding_corpus_path: str = 'feat_extractors/model_dim10/weights/embeddings/glove.840B.300d.txt',
            **kwargs
    ):
        """
        读取模型 if 已经训练过；训练模型 if not 训练过
        """
        # if self.trained():
        #     return

        ## Train and save embeddings
        if (not os.path.exists(os.path.join(self.embedding_dir, "glove.840B.300d.filtered.wv.vectors.npy")) or
                not os.path.exists(os.path.join(self.embedding_dir, "glove.840B.300d.filtered.wv"))
        ):
            log.info("Loading embedding vectors")
            glove_input_file = embedding_corpus_path
            word2vec_output_file = os.path.join(self.embedding_dir, "glove.840B.300d.vec")
            if not os.path.exists(word2vec_output_file):
                if not os.path.exists(glove_input_file):
                    raise FileNotFoundError(f"{glove_input_file} does not exist, download a copy of this corpus or trained glove.840B.300d.vec file from https://nlp.stanford.edu/projects/glove/")
                from gensim.scripts.glove2word2vec import glove2word2vec
                log.info("Training embedding vectors")
                glove2word2vec(glove_input_file, word2vec_output_file)
                log.info("Filtering embedified content")
                glove4gensim(word2vec_output_file)

        ## Initialize tokenizer and embedding module
        log.info("Initializing tokenizer and embedding layer")
        self.tokenizer = TweetTokenizer().tokenize
        self.embedding = ExtractWordEmbeddings(emb_type='glove', emb_dir=self.embedding_dir)  # time consuming

        ## Initialize dim10 models
        log.info("Initializing models for each dimension")
        self.models = []
        for dim_name in tqdm(self.dim_names, total=len(self.dim_names)):
            weight_file = os.path.join(self.lstm_model_dir, dim_name, "best-weights.pth")
            if not os.path.exists(weight_file):
                # Train dim 10 models
                raise NotImplementedError("TODO: train a dim10 extractor from scrach; but not needed for now...")  # TODO

            else:
                # Load dim 10 models
                model = LSTMClassifier(embedding_dim=300, hidden_dim=self.D).to(self.device)
                if weight_file is not None:
                    state = torch.load(weight_file, map_location=self.device)
                    model.load_state_dict(state)
            
            self.models.append(model)
        log.info(f"Initialized {len(self.models)} models")
    
    @torch.no_grad()
    def _predict_one_dim(self, model, em, tokenized_sents, batch_size, device):
        model.eval()
        scores = []
        n = len(tokenized_sents)
        for i in range(0, n, batch_size):
            batch_sents = tokenized_sents[i:i+batch_size]
            x = [em.obtain_vectors_from_sentence(sent, True) for sent in batch_sents]
            x = torch.tensor(padBatch(x)).float().to(device)
            out = model(x).detach().cpu().tolist()
            scores.extend(out)
        return scores
    
    def extract(self, text: Union[str, list[str]], **kwargs) -> NDArray:
        """
        Returns:
            NDArray shaped `[n, 10]`, where `n` is input text count
        """
        if not self.trained():
            log.info("Warning: dim10 extractor not trained yet, returning an empty array. Train dim10 extractor first")
            return np.empty(shape=(0, ))
        
        if isinstance(text, str):
            text = [text]
        
        tokenized = [self.tokenizer(preprocessText((t or "").lower())) for t in text]

        dims_scores = []
        for dim_name, model in zip(self.dim_names, self.models):
            scores = self._predict_one_dim(model, self.embedding, tokenized, self.B, self.device)
            try:
                scores = np.array([v.item() for v in scores], dtype=np.float32).reshape(-1, 1)
            except AttributeError:
                scores = np.array([v for v in scores], dtype=np.float32).reshape(-1, 1)
            dims_scores.append(scores)
        
        dims_scores = np.concat(dims_scores, axis=-1)  # [n, 10]
        return dims_scores