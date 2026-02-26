# coding=utf-8
"""
简化版的 ALBERT 预训练数据生成脚本
目的：理解数据预处理流程，特别是 Whole Word Masking (WWM)
已移除所有 TensorFlow 依赖，仅保留核心逻辑
"""

import argparse
import collections
import json
import random
import re

import jieba

# 导入本地的简化版 tokenization（无 TF 依赖）
import tokenization


class TrainingInstance(object):
    """单个训练实例（句子对）"""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([str(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([str(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

    def to_dict(self):
        """转换为字典格式，便于保存为JSON"""
        return {
            'tokens': self.tokens,
            'segment_ids': self.segment_ids,
            'is_random_next': self.is_random_next,
            'masked_lm_positions': self.masked_lm_positions,
            'masked_lm_labels': self.masked_lm_labels
        }


def get_new_segment(segment):
    """
    为中文全词遮蔽（WWM）添加标记

    输入一句话，返回一句经过处理的话：为了支持中文全词mask，将被分开的词加上特殊标记("##")，
    使得后续处理模块能够知道哪些字属于同一个词。

    Args:
        segment: 一句话，按字符分开的列表
                 例如: ['悬', '灸', '技', '术', '培', '训', '专', '家']

    Returns:
        处理过的句子列表
        例如: ['悬', '##灸', '技', '术', '培', '训', '专', '##家']

    原理：
    1. 使用 jieba 分词：['悬灸', '技术', '培训', '专家']
    2. 对于多字词，第一个字保持不变，后续字添加 ## 前缀
    3. 在后续的 mask 过程中，带 ## 的字会和前面的字作为一个整体被 mask
    """
    # 步骤1：使用 jieba 分词
    seq_cws = jieba.lcut("".join(segment))
    seq_cws_dict = {x: 1 for x in seq_cws}  # 转为字典方便查找

    new_segment = []
    i = 0

    # 步骤2：遍历每个字符
    while i < len(segment):
        # 如果不是中文字符，直接加入，不做特殊处理
        if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:
            new_segment.append(segment[i])
            i += 1
            continue

        # 步骤3：尝试匹配分词结果（从长到短，最长3个字）
        has_add = False
        for length in range(3, 0, -1):
            if i + length > len(segment):
                continue
            # 检查当前位置开始的 length 个字是否在分词结果中
            if ''.join(segment[i:i + length]) in seq_cws_dict:
                # 第一个字不加 ##
                new_segment.append(segment[i])
                # 后续的字都加上 ##
                for l in range(1, length):
                    new_segment.append('##' + segment[i + l])
                i += length
                has_add = True
                break

        # 如果没有匹配到任何词，单独加入这个字
        if not has_add:
            new_segment.append(segment[i])
            i += 1

    return new_segment


def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng, do_whole_word_mask=True):
    """从原始文本创建训练实例"""

    all_documents = [[]]

    # 读取输入文件
    # 格式：
    # (1) 每行一个句子
    # (2) 文档之间用空行分隔
    print("*** 读取输入文件 ***")
    for input_file in input_files:
        print(f"  {input_file}")
        with open(input_file, "r", encoding='utf-8') as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                # 空行表示文档分隔符
                if not line:
                    all_documents.append([])
                    continue

                # 对每行进行分词
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # 移除空文档
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    print(f"*** 共读取 {len(all_documents)} 个文档 ***")

    vocab_words = list(tokenizer.vocab.keys())
    instances = []

    # 创建多个副本（不同的mask）
    for dupe_idx in range(dupe_factor):
        print(f"*** 生成第 {dupe_idx + 1}/{dupe_factor} 轮数据 ***")
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document_albert(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng,
                    do_whole_word_mask))

    rng.shuffle(instances)
    print(f"*** 共生成 {len(instances)} 个训练实例 ***")
    return instances


def create_instances_from_document_albert(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng, do_whole_word_mask):
    """
    为单个文档创建训练实例
    简化版：只保留 MLM 任务，移除 SOP 任务
    """
    document = all_documents[document_index]

    # 预留 [CLS], [SEP] 的位置（只有一个句子）
    max_num_tokens = max_seq_length - 2

    # 目标序列长度（有一定概率使用较短的序列）
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    instances = []
    current_chunk = []  # 当前处理的文本段
    current_length = 0
    i = 0

    while i < len(document):
        segment = document[i]

        # ★★★ 核心：应用中文全词遮蔽（WWM）★★★
        segment = get_new_segment(segment)

        current_chunk.append(segment)
        current_length += len(segment)

        # 当达到目标长度或文档结束时，创建实例
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # 将当前块的所有句子合并成一个序列
                tokens = []
                for chunk in current_chunk:
                    tokens.extend(chunk)

                # 跳过空的情况
                if len(tokens) == 0:
                    i += 1
                    continue

                # 截断到最大长度
                if len(tokens) > max_num_tokens:
                    tokens = tokens[:max_num_tokens]

                # 组装成简化格式：[CLS] tokens [SEP]
                final_tokens = []
                segment_ids = []

                final_tokens.append("[CLS]")
                segment_ids.append(0)

                for token in tokens:
                    final_tokens.append(token)
                    segment_ids.append(0)

                final_tokens.append("[SEP]")
                segment_ids.append(0)

                # ★★★ 创建 Masked LM 的数据 ★★★
                (final_tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    final_tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, do_whole_word_mask)

                instance = TrainingInstance(
                    tokens=final_tokens,
                    segment_ids=segment_ids,
                    is_random_next=False,  # 不再使用 SOP 任务
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)

            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, do_whole_word_mask):
    """
    创建 Masked LM 的预测目标

    核心逻辑：
    1. 识别候选 token（跳过 [CLS] 和 [SEP]）
    2. 如果开启 WWM，将 ## 开头的 token 归入前一个 token 的集合
    3. 随机选择要 mask 的 token
    4. 80% 替换为 [MASK]，10% 保持不变，10% 替换为随机词
    """

    # 步骤1：构建候选索引
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue

        # ★★★ WWM 核心逻辑 ★★★
        # 如果当前 token 以 ## 开头，将其索引加入到前一个词的索引集合中
        # 这样在 mask 时会作为一个整体一起 mask
        if (do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    # 步骤2：随机打乱候选索引
    rng.shuffle(cand_indexes)

    # 步骤3：去掉中文字符前的 ## 标记（仅用于标识，实际 token 不需要）
    output_tokens = []
    for t in tokens:
        if len(re.findall('##[\u4E00-\u9FA5]', t)) > 0:
            output_tokens.append(t[2:])  # 去掉 ##
        else:
            output_tokens.append(t)

    # 步骤4：计算要 mask 的数量
    num_to_predict = min(max_predictions_per_seq,
                        max(1, int(round(len(tokens) * masked_lm_prob))))

    # 步骤5：执行 masking
    masked_lms = []
    covered_indexes = set()

    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break

        # 如果加上这个词会超过最大预测数量，跳过
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue

        # 检查是否已经被 mask
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue

        # Mask 整个词（可能是多个 token）
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% 的时间，替换为 [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% 的时间，保持原样
                if rng.random() < 0.5:
                    if len(re.findall('##[\u4E00-\u9FA5]', tokens[index])) > 0:
                        masked_token = tokens[index][2:]  # 去掉 ##
                    else:
                        masked_token = tokens[index]
                # 10% 的时间，替换为随机词
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def write_instances_to_json(instances, output_file, tokenizer, max_seq_length, max_predictions_per_seq):
    """将训练实例写入 JSON 文件（便于查看和理解）"""
    print(f"\n*** 写入输出文件: {output_file} ***")

    with open(output_file, 'w', encoding='utf-8') as writer:
        for inst_index, instance in enumerate(instances):
            # 转换为 ID
            input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
            input_mask = [1] * len(input_ids)
            segment_ids = list(instance.segment_ids)

            # Padding
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            masked_lm_positions = list(instance.masked_lm_positions)
            masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_ids)

            while len(masked_lm_positions) < max_predictions_per_seq:
                masked_lm_positions.append(0)
                masked_lm_ids.append(0)
                masked_lm_weights.append(0.0)

            features = {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_ids": masked_lm_ids,
                "masked_lm_weights": masked_lm_weights,
                # 为了便于理解，额外保存 tokens
                "tokens": instance.tokens,
                "masked_lm_labels": instance.masked_lm_labels
            }

            writer.write(json.dumps(features, ensure_ascii=False) + '\n')

            # 打印前几个样例
            if inst_index < 5:
                print(f"\n*** 样例 {inst_index + 1} ***")
                print("tokens:", " ".join(instance.tokens))
                print("segment_ids:", " ".join([str(x) for x in instance.segment_ids]))
                print("masked_lm_positions:", masked_lm_positions[:len(instance.masked_lm_positions)])
                print("masked_lm_labels:", " ".join(instance.masked_lm_labels))

    print(f"\n*** 共写入 {len(instances)} 个训练实例 ***")


def main():
    parser = argparse.ArgumentParser(description='简化版 ALBERT 预训练数据生成（理解 WWM 流程）')

    parser.add_argument('--input_file', type=str, required=True,
                       help='输入原始文本文件（逗号分隔多个文件）')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出 JSON 文件')
    parser.add_argument('--vocab_file', type=str, required=True,
                       help='词汇表文件')
    parser.add_argument('--do_lower_case', action='store_true', default=True,
                       help='是否转换为小写')
    parser.add_argument('--do_whole_word_mask', action='store_true', default=True,
                       help='是否使用全词遮蔽（WWM）')
    parser.add_argument('--max_seq_length', type=int, default=512,
                       help='最大序列长度')
    parser.add_argument('--max_predictions_per_seq', type=int, default=51,
                       help='每个序列的最大预测数')
    parser.add_argument('--random_seed', type=int, default=12345,
                       help='随机种子')
    parser.add_argument('--dupe_factor', type=int, default=10,
                       help='数据复制因子（不同的 mask）')
    parser.add_argument('--masked_lm_prob', type=float, default=0.15,
                       help='Masked LM 的概率')
    parser.add_argument('--short_seq_prob', type=float, default=0.1,
                       help='生成短序列的概率')

    args = parser.parse_args()

    print("=" * 80)
    print("简化版 ALBERT 预训练数据生成")
    print("目的：理解 Whole Word Masking (WWM) 流程")
    print("=" * 80)
    print("\n配置参数：")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    # 初始化 tokenizer
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    # 读取输入文件
    input_files = []
    for input_pattern in args.input_file.split(","):
        input_files.append(input_pattern)

    # 初始化随机数生成器
    rng = random.Random(args.random_seed)

    # 创建训练实例
    instances = create_training_instances(
        input_files, tokenizer, args.max_seq_length, args.dupe_factor,
        args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq,
        rng, args.do_whole_word_mask)

    # 写入输出文件
    write_instances_to_json(
        instances, args.output_file, tokenizer,
        args.max_seq_length, args.max_predictions_per_seq)

    print("\n✅ 处理完成！")


if __name__ == "__main__":
    main()

