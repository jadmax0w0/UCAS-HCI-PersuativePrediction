import json
import numpy as np
from numpy.typing import NDArray
from typing import Optional, Union, Literal

import sys; sys.path.append(".")
from feat_extractors.base_extractor import BaseTextFeatureExtractor
from utils import log
# import dashscope
# from dashscope import Generation

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
                log.info(f"Invalid id {id_str}, skipped this item")
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
                log.info(f"Key error at {group[0]['id']}:\n{ke}")
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
                log.info(f"Key error at \"{item['o_title']}\":\n{ke}")
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
        def call_llm_api(prompt: str, model_name: str = "qwen-turbo") -> str:
            try:
                response = Generation.call(
                    model=model_name,
                    prompt=prompt,
                    top_p=0.9,
                    temperature=0.3,
                )

                result_text = response.output["text"].strip()

                try:
                    _ = json.loads(result_text)
                except Exception:
                    raise ValueError(f"LLM 输出不是合法 JSON：{result_text}")

                return result_text
            
            except Exception as e:
                print(f"[LLM ERROR] {e}")

                return ((0,0,0,0,0,0), 
                        {
                    "Reciprocity": 0,
                    "Consistency": 0,
                    "Social_Proof": 0,
                    "Authority": 0,
                    "Scarcity": 0,
                    "Liking": 0
                })

        def build_cicero_prompt(text_p: str) -> str:
            definitions_en = {
                "Reciprocity": "The persuader provides something valuable (information, help, concession, or favor) creating a sense of obligation to reciprocate. This can be explicit ('I'll help you with X if you consider Y') or implicit.",
                "Consistency": "Referencing the target's previous statements, commitments, or values to highlight alignment or inconsistency. Includes appeals to their past behavior, stated beliefs, or core values.",
                "Social_Proof": "Citing majority opinion, social norms, statistics about what 'most people' do, testimonials, popularity, or group behavior to support the argument. This can be direct or indirect references to what others think/do.",
                "Authority": "Citing experts, scholars, official data, research findings, credentials, or personal/professional experience to establish credibility. This includes both explicit appeals to authority and implicit displays of expertise.",
                "Scarcity": "Emphasizing limited availability, uniqueness, time pressure, exclusivity, or potential loss/opportunity cost for not changing the view. This creates a sense of urgency or special opportunity.",
                "Liking": "Building rapport, similarity, or positive affect through compliments, empathy, friendly/humorous tone, shared experiences, or efforts to be agreeable. This makes the persuader more likable and thus more persuasive."
            }

            # 系统指令
            system_instruction = "You are an expert in persuasive language analysis. Identify which of Cialdini's Six Principles are used in a given text. Use these definitions:\n"
            for k, v in definitions_en.items():
                system_instruction += f"- {k}: {v}\n"

            # few-shot 
            few_shot_examples = '''EXAMPLES (Learn from these, but in your actual analysis, output ONLY JSON):

        Text: "I noticed you mentioned you value honesty. Given that, you might find this study from Harvard researchers interesting - it shows most people actually support transparency policies like the one I'm suggesting."
        [INTERNAL THINKING: Reciprocity=1 (offering study), Consistency=1 (referencing their value), Social_Proof=1 ("most people"), Authority=1 ("Harvard"), Scarcity=0 (none), Liking=1 (positive tone)]
        Output: {"Reciprocity":1, "Consistency":1, "Social_Proof":1, "Authority":1, "Scarcity":0, "Liking":1}

        Text: "As someone who's worked in this field for 20 years, I can tell you this opportunity won't last long. Many of my colleagues have already adopted this approach with great success."
        [INTERNAL THINKING: Reciprocity=0 (no offer), Consistency=0 (no reference to past), Social_Proof=1 ("many colleagues"), Authority=1 ("20 years experience"), Scarcity=1 ("won't last long"), Liking=0 (neutral)]
        Output: {"Reciprocity":0, "Consistency":0, "Social_Proof":1, "Authority":1, "Scarcity":1, "Liking":0}

        Text: "Hey, I really appreciate your thoughtful post! I used to think similarly until I saw the latest Pew Research data showing 80% support. Since you seem open-minded, I wanted to share this before the debate closes tomorrow."
        [INTERNAL THINKING: Reciprocity=1 (sharing data), Consistency=1 (appealing to open-mindedness), Social_Proof=1 ("80% support"), Authority=1 ("Pew Research"), Scarcity=1 ("before debate closes"), Liking=1 (compliment + friendly)]
        Output: {"Reciprocity":1, "Consistency":1, "Social_Proof":1, "Authority":1, "Scarcity":1, "Liking":1}
        '''

            # 用户指令
            user_instruction = f"""NOW ANALYZE THIS TEXT (Output ONLY JSON, no other text):

        Text: "{text_p}"

        Remember:
        1. Think about each principle internally
        2. Be generous - mark as 1 if there's ANY evidence
        3. Output ONLY the JSON object

        JSON Output:"""

            full_prompt = f"{system_instruction}\n{few_shot_examples}\n{user_instruction}"
            return full_prompt

        prompt = build_cicero_prompt(text)
        json_response = call_llm_api(prompt)
        return json_response
