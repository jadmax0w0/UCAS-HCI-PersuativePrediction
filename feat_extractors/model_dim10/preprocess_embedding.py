import gensim
from gensim.models import KeyedVectors
import os

def preprocess_line(line):
    """
    过滤无效的行
    :param line: 单行数据
    :return: 如果是有效行则返回 True，否则返回 False
    """
    # 检查是否包含邮箱或无效字符
    if '@' in line or '.' not in line:
        return False
    
    # 确保每一行以正确的格式出现
    try:
        # 尝试分割行并检查是否包含有效的数字
        parts = line.split()
        if len(parts) > 1:  # 忽略单一的无用字符行
            # 判断第一项是否为数字
            float(parts[0])
            return True
    except ValueError:
        return False
    
    return False

def filter_lines(file_path):
    """
    过滤掉包含无效数据的行，避免错误发生
    :param file_path: 输入的文件路径
    :return: None
    """
    valid_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if preprocess_line(line):
                valid_lines.append(line)
    
    # 计算词汇大小和向量维度
    vocab_size = len(valid_lines)
    vector_size = len(valid_lines[0].split()) - 1  # 词向量大小为每行的元素数量减去1（去掉词本身）
    
    # 在文件开头添加头部信息
    header = f"{vocab_size} {vector_size}\n"
    
    # 将过滤后的行写入新的文件
    filtered_file_path = file_path.replace('.vec', '.filtered.vec')
    with open(filtered_file_path, 'w', encoding='utf-8') as f:
        f.write(header)  # 添加头部
        f.writelines(valid_lines)  # 写入有效行
    
    print(f"Filtered file saved to {filtered_file_path}")
    return filtered_file_path

def glove4gensim(file_dir):
    """
    修改预训练的GloVe文件，使其可以与此框架集成
    :param file_dir: 输入的GloVe文件路径
    :return: None
    """
    from gensim.scripts.glove2word2vec import glove2word2vec

    # 先过滤掉无效行
    filtered_file_path = filter_lines(file_dir)
    
    # 使用过滤后的文件进行处理
    assert filtered_file_path.endswith('.vec'), "Input file should be .vec"

    # 加载并转换GloVe文件为Word2Vec格式
    model = KeyedVectors.load_word2vec_format(filtered_file_path, binary=False)

    # 保存词向量部分到 .wv 文件
    new_file_dir = filtered_file_path.replace('.vec', '.wv')
    model.save(new_file_dir)  # 使用模型的save方法保存词向量

    # 删除原始的 .vec 文件
    os.remove(filtered_file_path)
    print(f"Removed previous file {filtered_file_path}")

    # 尝试加载新的 .wv 文件
    model = KeyedVectors.load(new_file_dir)
    
    # 修改后的访问方式，使用key_to_index替代vocab
    print(f"Loaded in gensim! {len(model.key_to_index)} word embeddings, {len(model[model.index_to_key[0]])} dimensions")
    return

if __name__ == '__main__':
    # 使用已生成的GloVe文件路径
    glove4gensim('weights/embeddings/glove.840B.300d.vec')  # 将路径替换为你的文件路径
