import json
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Literal

import sys; sys.path.append(".")
from feat_extractors.base_extractor import BaseTextFeatureExtractor


def load_from_anno_file(extracted_feat_path: str, partition: Optional[Literal['train', 'val']] = 'train'):
    """
    Returns:
        tuple: (P_matrix shaped `[N_p, 6]`, N_matrix shaped `[N_n, 6]`); expected `N_p == N_n`
    """
    SCORE_KEYS = ['Reciprocity', 'Consistency', 'Social_Proof', 'Authority', 'Scarcity', 'Liking']
    
    with open(extracted_feat_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    ## Cialdini scores for training set
    if partition == "train":
        json_data = json_data['extracted_scores']

        groups = []
        curr_group = []
        curr_prefix = None
        curr_main_id = None

        for item in json_data:
            id_str = item['id']
            try:
                part1, _ = id_str.split('_')
                prefix = part1[0]  # "P"/"N"
                main_id = part1[1:]
            except ValueError:
                print(f"Invalid id {id_str}, skipped this item")
                continue

            if not curr_group:
                curr_group.append(item)
                curr_prefix = prefix
                curr_main_id = main_id
            else:
                # If new item belongs to the current group...
                if prefix == curr_prefix and main_id == curr_main_id:
                    curr_group.append(item)
                else:
                    # ...otherwise create a new group
                    groups.append(curr_group)
                    curr_group = [item]
                    curr_prefix = prefix
                    curr_main_id = main_id
        
        if curr_group:
            groups.append(curr_group)

        p_arrays = []
        n_arrays = []

        for group in groups:
            try:
                # Get prefix ("P"/"N")
                group_prefix = group[0]['id'].split('_')[0][0]

                # Get the scores for each item in this group
                group_score_arrays = []
                for item in group:
                    scores = item['scores']
                    score_array = np.array([scores[key] for key in SCORE_KEYS], dtype=np.int32)
                    group_score_arrays.append(score_array)
                
                # Merge score arrays in this group (by max-pooling)
                merged_array = np.max(np.stack(group_score_arrays), axis=0)

                if group_prefix == 'P':
                    p_arrays.append(merged_array)
                elif group_prefix == 'N':
                    n_arrays.append(merged_array)
            
            except KeyError as ke:
                print(f"Key error at {group[0]['id']}:\n{ke}")
                continue

        P_matrix = np.stack(p_arrays) if p_arrays else np.empty((0, 6), dtype=np.int32)
        N_matrix = np.stack(n_arrays) if n_arrays else np.empty((0, 6), dtype=np.int32)

        return P_matrix, N_matrix

    ## Cialdini scores for validation set
    elif partition == "val":
        p_arrays = []
        n_arrays = []

        for item in json_data:
            try:
                curr_p_scores = item['p_positive_c_scores']
                curr_n_scores = item['p_negative_c_scores']

                curr_p_arrays = np.array([[scores[k] for k in SCORE_KEYS] for scores in curr_p_scores], dtype=np.int32)
                curr_n_arrays = np.array([[scores[k] for k in SCORE_KEYS] for scores in curr_n_scores], dtype=np.int32)

                curr_p_array = curr_p_arrays.max(axis=0)
                curr_n_array = curr_n_arrays.max(axis=0)

                p_arrays.append(curr_p_array)
                n_arrays.append(curr_n_array)
            
            except KeyError as ke:
                print(f"Key error at \"{item['o_title']}\":\n{ke}")
                continue
        
        P_matrix = np.stack(p_arrays, axis=0) if p_arrays else np.empty((0, 6), dtype=np.int32)
        N_matrix = np.stack(n_arrays, axis=0) if n_arrays else np.empty((0, 6), dtype=np.int32)

        return P_matrix, N_matrix

    else:
        raise NotImplementedError(f"Unknown partition {partition} while loading Cialdini scores from annotation file {extracted_feat_path}")


class CialdiniFeatureExtractor(BaseTextFeatureExtractor):
    def __init__(self):
        super().__init__()
    
    def trained(self):
        return True
    
    def train(self, **kwargs):
        return
    
    def extract(self, text: Union[str, list[str]], **kwargs) -> NDArray:
        # TODO: 运行时调用 qwen api
        raise NotImplementedError()
