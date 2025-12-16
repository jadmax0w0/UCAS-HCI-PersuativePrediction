import torch
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional, Union
from transformers import BertTokenizer, BertModel

import sys; sys.path.append(".")
from feat_extractors.base_extractor import BaseTextFeatureExtractor
from utils import log


class BertTextFeatureExtractor(BaseTextFeatureExtractor):
    def __init__(self, model_path: str = "bert-base-multilingual-cased", minibatch_size: Optional[int] = 128):
        super().__init__()
        self.model_path = model_path

        self.tokenizer = None
        self.model = None
        self.device = "cpu"

        self.minibatch_size = minibatch_size
    
    def trained(self):
        return self.tokenizer is not None and self.model is not None
    
    def lazy_initialization(self):
        log.info("Loading Bert model")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertModel.from_pretrained(self.model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to(device=self.device)
    
    def train(self, *args, **kwargs):
        """Alias for `BertTextFeatureExtractor.lazy_initialization()`"""
        self.lazy_initialization()
    
    def extract(self, text: Union[str, list[str]], stride: int = 10, max_length: int = 512, **kwargs) -> NDArray:
        """
        Args:
            stride: 滑动窗口步长，0表示不滑动
            max_length: 最大长度
        Returns:
            NDArray shaped `[n, D_bert]`, where `n` is input text count
        """
        if isinstance(text, str):
            text = [text]

        def process_long_text(single_text: str):
            tokens = self.tokenizer.encode(single_text, truncation=False)
            if len(tokens) <= max_length:
                encoded_input = self.tokenizer(
                    single_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                encoded_input = encoded_input.to(device=self.device)
                output = self.model(**encoded_input)
                return output["pooler_output"].detach().cpu().numpy()

            features = []
            for i in range(0, len(tokens), max_length - stride):
                window_tokens = tokens[i:i + max_length]
                window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
                encoded_input = self.tokenizer(
                    window_text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                encoded_input = encoded_input.to(device=self.device)
                window_output = self.model(**encoded_input)
                window_feat = window_output["pooler_output"].detach().cpu().numpy()

                features.append(window_feat)

            if features:
                return np.mean(features, axis=0)
            else:
                return np.zeros((1, self.model.config.hidden_size))

        if self.minibatch_size is None:
            if stride > 0:
                output_list = []
                for single_text in text:
                    feat = process_long_text(single_text)
                    output_list.append(feat)
                output = np.concatenate(output_list, axis=0)
            else:
                encoded_input = self.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,  # <<< 显式启用截断
                    max_length=max_length  # <<< 显式设置最大长度
                )
                encoded_input = encoded_input.to(device=self.device)
                model_output = self.model(**encoded_input)
                output = model_output["pooler_output"].detach().cpu().numpy()

        else:
            output = []
            for s in tqdm(range(0, len(text), self.minibatch_size),
                          desc="Exporting feats via Bert",
                          total=int(np.ceil(len(text) / self.minibatch_size).item())):
                t = min(len(text), s + self.minibatch_size)
                mb_text = text[s:t]

                if stride > 0:
                    mb_features = []
                    for single_text in mb_text:
                        feat = process_long_text(single_text)
                        mb_features.append(feat)
                    mb_output_np = np.concatenate(mb_features, axis=0)
                else:
                    encoded_input = self.tokenizer(
                        mb_text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    )
                    encoded_input = encoded_input.to(device=self.device)
                    mb_output = self.model(**encoded_input)
                    mb_output_np = mb_output["pooler_output"].detach().cpu().numpy()

                output.append(mb_output_np)
                if stride <= 0:
                    del mb_output
            output = np.concatenate(output, axis=0)
        return output
