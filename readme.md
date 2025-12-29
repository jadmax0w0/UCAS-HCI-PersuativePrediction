# X16-Persuader 可解释说服性预测

## 数据集

- 训练集：`dataset/train.jsonl`
- 验证集（同样用于测试模型性能）：`dataset/val.jsonl`
- 训练集的 Cialdini 6 维分数：`dataset/cialdini_scores_train.json`
- 验证集的 Cialdini 6 维分数：`dataset/cialdini_scores_val.json`

数据集详细格式详见结题报告 slides; Cialdini 6 维分数数据集的读取代码为 `feat_extractors.cialdini_extractor.load_from_anno_file()` 函数。

## 代码运行

### 训练 + 预测

通过 `pip install -r requirements.txt` 配置好环境之后，直接使用下面的命令便可以启动模型的训练 & 预测：

```bash
python main.py
```

**自定义项：**

可以在 main.py 中修改部分内容实现自定义项：

- `bert_ext = BertTextFeatureExtractor("/path/to/bert/ckpt", minibatch_size=128)` 中，可以自定义 Bert 模型的检查点
- `train_model(...)` 函数的调用中，可以通过 `model_save_path` 自定义模型保存名称与路径（注意：只保存最终的 classifier 模型部分，并不会保存几个 feature extractor）

另外，main.py 仅支持训练 + 预测一体化。如果未修改 `model_save_path`，那么两次运行 main.py 将会覆盖模型文件。

### 仅预测

要想通过训练好的模型文件只进行预测，可以通过 inference.py 进行：

```bash
python inference.py
```

**自定义项：**

- 同样可以在 `bert_ext` 行中自定义 Bert 模型的检查点路径
- 可以在 `model = load_model("model/test2.pkl")` 行中自定义 classifier 的路径
- 可以在全局变量的 scope 内使用 `O`, `P` 变量来指定自定义的说服性预测内容（inference.py 代码文件中已经提供了一些注释掉的样例）
  - 注意：由于负责导出 10 维特征的模型并没有在中文语料上训练过，因此如果要针对中文文本使用，在不考虑性能问题的情况下可以注释掉 `p_feat = get_custom_text_features(...)` 调用中的 `dim10_extractor=dm10_ext,`，显式指定不使用 10 维模型。对于英文文本，则可以解注释。
  - 由于 Cialdini 6 维特征通过大模型标注得到，因此可以将任意语言的待预测文本使用我们提供的标注代码标注好。

### 可视化界面

通过 `python app.py` 即可开启可视化界面。可视化界面以网页形式呈现。

注意：

- 在运行该代码前，需要填写其中的模型检查点路径。填写示例如下所示（对应 app.py 中的代码）：

```python
try:
    model = load_model("path/to/model")
except Exception as e:
    print(f"Load model failed. Locate your model checkpoint first.\n{e}")
```

- 其余注意事项可以参考 [仅预测](#仅预测)（*注：我们在该部分已经实现了自动化调用 LLM 获取 Cialdini 6 维评分的代码，所以无需调用标注代码另行标注目标文本的 6 维分数*）。
