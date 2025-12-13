import torch
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional, Union
from transformers import BertTokenizer, BertModel

import sys; sys.path.append(".")
from feat_extractors.base_extractor import BaseTextFeatureExtractor


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
        print("Loading Bert model")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertModel.from_pretrained(self.model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = "cuda"
            self.model = self.model.to(device=self.device)
    
    def train(self, *args, **kwargs):
        """Alias for `BertTextFeatureExtractor.lazy_initialization()`"""
        self.lazy_initialization()
    
    def extract(self, text: Union[str, list[str]], **kwargs) -> NDArray:
        """
        Returns:
            NDArray shaped `[n, D_bert]`, where `n` is input text count
        """
        if isinstance(text, str):
            text = [text]
        
        if self.minibatch_size is None:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True)
            encoded_input = encoded_input.to(device=self.device)
            output = self.model(**encoded_input)
            output = output["pooler_output"].detach().cpu().numpy()
        
        else:
            output = []
            for s in tqdm(range(0, len(text), self.minibatch_size), desc="Exporting feats via Bert", total=int(np.ceil(len(text) / self.minibatch_size).item())):
                t = min(len(text), s + self.minibatch_size)
                mb_text = text[s:t]

                encoded_input = self.tokenizer(mb_text, return_tensors='pt', padding=True)
                encoded_input = encoded_input.to(device=self.device)
                mb_output = self.model(**encoded_input)
                mb_output_np = mb_output["pooler_output"].detach().cpu().numpy()
                del mb_output
                output.append(mb_output_np)
            output = np.concat(output, axis=0)

        return output
