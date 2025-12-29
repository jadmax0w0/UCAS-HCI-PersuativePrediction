import gradio as gr
import plotly.graph_objects as go
import random
import pandas as pd


import numpy as np

import sys; sys.path.append(".")
from feat_extractors.cialdini_extractor import CialdiniFeatureExtractor
from feat_extractors.dim10_extractor import Dim10FeatureExtractor
from feat_extractors.bert_extractor import BertTextFeatureExtractor
from classifiers.bagger.bagger_presets import get_bagger_model
from classifiers.bagger.voting_bagger import VotingBagger
from pipeline import train_model, eval_model, load_model, get_custom_text_features


# ==========================================
# 1. 常量定义
# ==========================================


O = """我认为人类是没有自由意志的，这一观点可以从经典的神经科学实验中找到有力支撑。早有研究发现，人在实际做出决定与意识到自己做了决定之间，存在一段短则几百毫秒、长则数秒的时间差 —— 当我们主观上觉得 “是自己主动选择做某件事” 时，大脑其实早已提前形成了对应的神经活动模式，完成了决策的核心过程。我们所感知到的 “自主决策” 的体验，本质上更像是大脑为了让我们理解自身行为，事后构建出的一种认知幻觉，而非真正由 “自我意识” 主导的主动选择。这意味着从神经生物学的底层逻辑来看，所谓的自由意志或许并非人类意识的自主产物，而是大脑神经活动的附带结果。
"""

P = """
首先，该结论过度简化了神经实验的场景与结论。支撑 “无自由意志” 的经典实验（如利贝特实验），大多是让受试者做无意义的、瞬间的简单决策（比如随机按左键或右键），这类决策依赖的是大脑的本能反应或随机神经活动，和现实中人类的复杂决策完全不是同一维度。当我们面临 “是否要换一份工作”“要不要帮陌生人一把” 这类需要权衡价值观、道德准则、长期利益的复杂选择时，大脑的活动模式会涉及前额叶皮层的理性分析、记忆调取、情绪整合等多个复杂模块，而非实验中那种单一的 “神经准备电位”。实验中几百毫秒的时延，更可能是大脑对简单动作的预备活动，而非对 “决策本身” 的提前定调，不能用简单场景的结论推导复杂的人类行为。

其次，混淆了 “神经活动的提前发生” 与 “决策的不可逆性”。实验中观测到的 “决策前的神经活动”，未必是 “最终决策的定论”，更可能是大脑的 “预选与筹备” 状态 。比如，当你纠结要不要喝奶茶时，大脑可能同时激活 “想喝” 和 “怕胖” 两个神经通路，此时出现的神经活动只是备选方案的预演；而最终你决定 “不喝”，很可能是意识主动介入，压制了 “想喝” 的神经冲动 —— 这个干预过程，恰恰体现了意识对决策的主导性。如果决策真的是大脑提前设定好的 “幻觉”，我们就无法解释 “临时改主意” 这种普遍现象。

第三，忽视了意识的 “主动建构与反馈能力”。自由意志的核心并非 “决策比意识早几百毫秒”，而是 “人类能根据自我认知、外部反馈调整后续行为”。比如，一个人第一次偷东西时，可能是本能的欲望驱动，但当他意识到这个行为的道德后果和法律风险后，后续能主动克制自己的欲望 —— 这种 “基于反思的行为修正”，是单纯的神经冲动无法解释的。大脑的神经活动和意识之间，不是 “单向的决定关系”，而是双向的互动关系：意识可以通过学习、反思重塑神经通路，反过来影响未来的决策；这种 “自我塑造” 的能力，正是自由意志的核心体现。

最后，对 “自由意志” 的定义陷入了 “非黑即白” 的误区。很多持 “无自由意志” 观点的人，默认自由意志是 “完全脱离物理规律、不受任何因果约束的绝对自由”—— 但这是一种不切实际的定义。从哲学和科学的共识来看，更合理的自由意志定义是 “能够基于自身的意愿、信念和理性，自主做出选择并为选择负责的能力”。即使大脑的决策有神经活动的基础（符合物理因果律），也不影响 “自由意志” 的存在 —— 就像电脑的运行依赖电路和代码，但我们依然会说 “程序员自由地编写了软件”，而非 “代码决定了一切”。人类的决策基于神经活动，但神经活动本身是由我们的经历、思考、价值观塑造的，这种 “自我决定的因果链”，正是自由意志的本质。
"""


# 西奥迪尼影响力法则 6维
CIALDINI_DIMS = [
    "互惠性 (Reciprocity)", "承诺与一致性 (Consistency)", 
    "社会认同 (Social Proof)", "权威 (Authority)", 
    "喜好 (Liking)", "稀缺性 (Scarcity)"
]

# 社媒分析数据 10维 (标题已更新)
SOCIAL_TITLE_EN = "TEN social dimensions of conversations and relationships"
SOCIAL_DIMS = [
    "知识 (Knowledge)", "权力 (Power)", "地位 (Status)", 
    "信任 (Trust)", "支持 (Support)", "浪漫 (Romance)", 
    "相似性 (Similarity)", "身份 (Identity)", "趣味 (Fun)", "冲突 (Conflict)"
]

# 维度解释文本
EXPLANATIONS = {
    "西奥迪尼6维法则": {
        "互惠性": "人们倾向于回报他人的恩惠。",
        "承诺与一致性": "人们倾向于遵守公开的承诺。",
        "社会认同": "人们倾向于跟随大众的选择。",
        "权威": "人们倾向于服从权威专家的意见。",
        "喜好": "人们倾向于答应自己喜欢的人的请求。",
        "稀缺性": "越稀缺的东西，人们越觉得有价值。"
    },
    SOCIAL_TITLE_EN: {
        "知识": "文本中体现的信息量或专业度。",
        "权力": "文本中体现的控制力或支配感。",
        "地位": "说话者在社交层级中的相对位置。",
        "信任": "文本传递的安全感与可靠性。",
        "支持": "文本表达的情感支持或赞同。",
        "浪漫": "涉及情感、恋爱或亲密关系的表达。",
        "相似性": "强调说话者与受众的共同点。",
        "身份": "关于自我认同或群体归属的表达。",
        "趣味": "幽默、娱乐或轻松的元素。",
        "冲突": "分歧、争论或对立的情绪。"
    }
}

# ==========================================
# 2. 核心模型接口
# ==========================================

def get_prediction(user_text, persuasion_text):
    
    # model = get_bagger_model(enable_xgb=False)
    try:
        model = load_model("path/to/model")
    except Exception as e:
        print(f"Load model failed. Locate your model checkpoint first.\n{e}")

    bert_ext = BertTextFeatureExtractor("/Users/youxseem/Documents/AIModels.localized/bert-base-multilingual-cased", minibatch_size=128)
    cial_ext = CialdiniFeatureExtractor()  # TODO: cialdini extractor WIP
    dm10_ext = Dim10FeatureExtractor()

    # model = load_model("model/test2.pkl")
    assert isinstance(model, VotingBagger)
    bert_ext.train()
    cial_ext.train()
    dm10_ext.train()

    # o = input("Opinion:\n")
    # p = input("Persuasive:\n")
    # cial = input("Cialdini (split with comma)\n[Reciprocity, Consistency, Social_Proof, Authority, Scarcity, Liking]:\n")
    # cial = np.array([int(v) for v in cial.split(",")]).reshape(1, -1)
    # cial = np.array([[0,0,0,0,1,0]])
    
    o = user_text
    p = persuasion_text
    o_feat = get_custom_text_features(
        text=o,
        bert_extractor=bert_ext,
    )
    p_feat = get_custom_text_features(
        text=p,
        cialdini_extractor=cial_ext,
        dim10_extractor=dm10_ext,
        bert_extractor=bert_ext,
    )

    cialdini_scores = p_feat[0,:6].tolist()
    social_scores = p_feat[0,6:16].tolist()
    
    feat = np.concat([p_feat, o_feat], axis=-1)
    # print(f"{feat.shape=}")
    y_pred = model.predict(feat)
    # print(f"{y_pred=}")
    
    return y_pred, cialdini_scores, social_scores

# ==========================================
# 3. 数据处理与可视化逻辑
# ==========================================

def generate_charts_and_result(user_text, persuasion_text):
    if not user_text or not persuasion_text:
        return "⚠️ 请输入完整文本", "", None, None

    # 1. 调用模型接口
    pred_label, c6_scores, s10_scores = get_prediction(user_text, persuasion_text)
    
    # 2. 处理预测结果文本
    result_str = "✅ 预测结果：可说服" if pred_label >= 0.5 else "❌ 预测结果：不可说服"
    
    # 3. 计算核心特征 Top 3
    all_features = {}
    for i, name in enumerate(CIALDINI_DIMS):
        all_features[name] = c6_scores[i]
    for i, name in enumerate(SOCIAL_DIMS):
        all_features[name] = s10_scores[i]
    print("All features:")
    print(all_features)
        
    sorted_features = sorted(all_features.items(), key=lambda item: item[1], reverse=True)[:3]
    top3_md = "### 🔥 核心驱动特征 Top 3\n"
    for rank, (name, score) in enumerate(sorted_features, 1):
        # 分数保留两位小数
        top3_md += f"{rank}. **{name}**: {score:.2f}\n"

    # 动画设置
    animation_settings = {
        'duration': 800,       
        'easing': 'cubic-out'  
    }

    # 4. 绘制图表 1：西奥迪尼 6维 (水平条形图)
    fig_bar = go.Figure(go.Bar(
        x=c6_scores,
        y=CIALDINI_DIMS,
        orientation='h',
        marker=dict(color='rgba(50, 171, 96, 0.7)', line=dict(color='rgba(50, 171, 96, 1.0)', width=1)),
        text=c6_scores,
        textposition='auto'
    ))
    fig_bar.update_layout(
        title="西奥迪尼影响力法则 (6维)",
        # ⚠️ 修改：X轴范围约束为 [0, 1]
        xaxis=dict(range=[0, 1], fixedrange=True), 
        yaxis=dict(fixedrange=True),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        transition=animation_settings
    )

    # 5. 绘制图表 2：TEN social dimensions (雷达图)
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=s10_scores + [s10_scores[0]], 
        theta=SOCIAL_DIMS + [SOCIAL_DIMS[0]], 
        fill='toself',
        line_color='deepskyblue',
        mode='lines+markers',
        marker=dict(size=5)
    ))
    fig_radar.update_layout(
        # ⚠️ 修改：标题更新为英文标题
        title=SOCIAL_TITLE_EN,
        polar=dict(
            # ⚠️ 修改：径向轴范围约束为 [0, 1]
            radialaxis=dict(visible=True, range=[0, 1]),
            angularaxis=dict() 
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=400,
        transition=animation_settings
    )

    return result_str, top3_md, fig_bar, fig_radar

# ==========================================
# 4. Gradio 界面构建
# ==========================================

def create_ui():
    with gr.Blocks(title="说服预测模型可解释性分析", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("## 🧠 说服预测模型可视化分析平台\n输入用户原文与说服文本，分析说服成功率及背后的心理学/社会学特征。")
        
        with gr.Row():
            with gr.Column():
                input_user = gr.Textbox(
                    label="用户原文 (User Original Text)", 
                    lines=5, 
                    placeholder="请输入用户原始表达的观点或需求...",
                    value=O#"First let me say that I know a better way to help the homeless is volunteering, serving meals, etc, and I do that. But when a homeless person approaches me specifically and asks for money, it pains me to say no. Often times I have seen the same person I just refused in front of a store go buy something rather extravagant/unnecessary for someone who is begging for money. I have also seen them pull out their iPhone immediately after. I know not all homeless people are like this, but it seems to be a lot. I also know that they have mental disorders, rough home lives with no family, addictions, and no way to be presentable for a job. But if I give to something, I want to know where it's going and what it's used for. With giving directly to the homeless, this isn't there. Change my view."
                )
            with gr.Column():
                input_persuasion = gr.Textbox(
                    label="说服文本 (Persuasion Text)", 
                    lines=5, 
                    placeholder="请输入尝试说服用户的文本...",
                    value=P#"Just because someone has an iphone doesn't mean they don't need help.  I mean my parents could kick me out today and I lose my job but I'll still have an iphone.  "
                )
        
        btn_predict = gr.Button("🚀 开始预测 (Start Prediction)", variant="primary", scale=0)
        
        gr.Markdown("---")
        
        # 结果显示区
        with gr.Row():
            with gr.Column(scale=1):
                out_result = gr.Markdown("### 等待预测...", label="预测结论")
            with gr.Column(scale=1):
                out_top3 = gr.Markdown("", label="核心特征")
        
        # 可视化图表区
        with gr.Row():
            with gr.Column():
                plot_cialdini = gr.Plot(label="西奥迪尼6维分布")
            with gr.Column():
                # label 显示更新后的标题
                plot_social = gr.Plot(label=SOCIAL_TITLE_EN)
        
        # 维度说明折叠区
        with gr.Accordion("📚 点击查看特征维度详细定义", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 西奥迪尼影响力法则")
                    for k, v in EXPLANATIONS["西奥迪尼6维法则"].items():
                        gr.Markdown(f"- **{k}**: {v}")
                with gr.Column():
                    # 标题更新
                    gr.Markdown(f"#### {SOCIAL_TITLE_EN}")
                    for k, v in EXPLANATIONS[SOCIAL_TITLE_EN].items():
                        gr.Markdown(f"- **{k}**: {v}")

        # 绑定事件
        btn_predict.click(
            fn=generate_charts_and_result,
            inputs=[input_user, input_persuasion],
            outputs=[out_result, out_top3, plot_cialdini, plot_social]
        )
        
    return demo

if __name__ == "__main__":
    app = create_ui()
    app.launch(inbrowser=True, share=False)