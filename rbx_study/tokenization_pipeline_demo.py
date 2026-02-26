# coding=utf-8
"""
详细演示 tokenization.py 的分词 pipeline

分词流程：
1. BasicTokenizer: 基础分词
   - convert_to_unicode: Unicode 转换
   - _clean_text: 清理文本（去除控制字符）
   - _tokenize_chinese_chars: 中文字符周围添加空格
   - whitespace_tokenize: 按空格分割
   - _run_strip_accents: 去除重音（可选）
   - _run_split_on_punc: 按标点分割

2. WordpieceTokenizer: WordPiece 分词
   - 将 BasicTokenizer 输出进一步切分为子词
   - 使用贪心最长匹配算法
   - 添加 ## 前缀表示子词

3. FullTokenizer: 完整流程
   - BasicTokenizer + WordpieceTokenizer
   - Token → ID 转换
"""

import tokenization


def show_step(step_name, content, description=""):
    """显示处理步骤"""
    print(f"\n{'='*80}")
    print(f"【步骤】{step_name}")
    if description:
        print(f"【说明】{description}")
    print(f"{'='*80}")
    print(f"结果: {repr(content)}")


def demo_basic_tokenizer_step_by_step(text):

    """逐步演示 BasicTokenizer 的处理流程"""
    print("\n" + "█" * 80)
    print("█  Part 1: BasicTokenizer 详细流程")
    print("█" * 80)

    show_step("原始输入", text)

    # Step 1: Unicode 转换
    text_unicode = tokenization.convert_to_unicode(text)
    show_step("1. convert_to_unicode", text_unicode,
              "确保文本是 Unicode 格式")

    # Step 2: 清理文本
    tokenizer = tokenization.BasicTokenizer(do_lower_case=True)
    text_cleaned = tokenizer._clean_text(text_unicode)
    show_step("2. _clean_text", text_cleaned,
              "去除控制字符（0、0xfffd）、将所有空白字符统一为空格")

    # Step 3: 中文字符周围添加空格
    text_with_space = tokenizer._tokenize_chinese_chars(text_cleaned)
    show_step("3. _tokenize_chinese_chars", text_with_space,
              "在每个中文字符前后添加空格，方便后续分词")

    # Step 4: 按空格初步分割
    orig_tokens = tokenization.whitespace_tokenize(text_with_space)
    show_step("4. whitespace_tokenize (第1次)", orig_tokens,
              "按空格分割，得到初步的 token 列表")

    # Step 5: 小写化 + 去重音
    split_tokens = []
    for i, token in enumerate(orig_tokens):
        token_lower = token.lower() if tokenizer.do_lower_case else token
        token_no_accent = tokenizer._run_strip_accents(token_lower) if tokenizer.do_lower_case else token_lower
        print(f"\n  Token #{i}: '{token}' → lower: '{token_lower}' → no_accent: '{token_no_accent}'")

        # Step 6: 按标点分割
        punc_split = tokenizer._run_split_on_punc(token_no_accent)
        print(f"           → split_on_punc: {punc_split}")
        split_tokens.extend(punc_split)

    show_step("5-6. 小写化 + 去重音 + 标点分割", split_tokens,
              "对每个 token 小写化、去重音、按标点符号分割")

    # Step 7: 再次按空格分割（清理）
    output_tokens = tokenization.whitespace_tokenize(" ".join(split_tokens))
    show_step("7. whitespace_tokenize (第2次)", output_tokens,
              "再次按空格分割，得到最终的 BasicTokenizer 输出")

    return output_tokens


def demo_wordpiece_tokenizer(basic_tokens, vocab_file):
    """演示 WordpieceTokenizer 的处理流程"""
    print("\n" + "█" * 80)
    print("█  Part 2: WordpieceTokenizer 详细流程")
    print("█" * 80)

    show_step("输入 (来自 BasicTokenizer)", basic_tokens,
              "BasicTokenizer 的输出作为 WordpieceTokenizer 的输入")

    vocab = tokenization.load_vocab(vocab_file)
    wordpiece_tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)

    print(f"\n词表大小: {len(vocab)}")
    print(f"前10个词: {list(vocab.keys())[:10]}")

    all_output_tokens = []
    for i, token in enumerate(basic_tokens):
        print(f"\n{'-'*80}")
        print(f"处理 Token #{i}: '{token}'")
        print(f"{'-'*80}")

        # 模拟 WordPiece 贪心匹配
        chars = list(token)
        print(f"  字符列表: {chars}")

        if len(chars) > wordpiece_tokenizer.max_input_chars_per_word:
            print(f"  ⚠ 超过最大长度 ({wordpiece_tokenizer.max_input_chars_per_word})，标记为 [UNK]")
            all_output_tokens.append("[UNK]")
            continue

        start = 0
        sub_tokens = []
        print(f"\n  贪心最长匹配过程:")

        while start < len(chars):
            end = len(chars)
            cur_substr = None

            # 从最长子串开始尝试
            attempts = []
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr

                in_vocab = substr in vocab
                attempts.append((substr, in_vocab))

                if in_vocab:
                    cur_substr = substr
                    break
                end -= 1

            # 显示匹配过程
            print(f"    位置 {start}: 尝试匹配")
            for attempt_str, found in attempts[:5]:  # 只显示前5次尝试
                status = "✓ 找到" if found else "✗ 不在词表"
                print(f"      '{attempt_str}': {status}")
                if found:
                    break
            if len(attempts) > 5:
                print(f"      ... (共尝试 {len(attempts)} 次)")

            if cur_substr is None:
                print(f"    ⚠ 无法匹配，整个 token 标记为 [UNK]")
                sub_tokens = ["[UNK]"]
                break
            else:
                print(f"    ✓ 匹配成功: '{cur_substr}'")
                sub_tokens.append(cur_substr)
                start = end

        print(f"\n  最终子词: {sub_tokens}")
        all_output_tokens.extend(sub_tokens)

    show_step("WordpieceTokenizer 输出", all_output_tokens,
              "## 前缀表示该子词不是词的开头")

    return all_output_tokens


def demo_full_pipeline(text, vocab_file):
    """演示完整的分词流程"""
    print("\n" + "█" * 80)
    print("█  Part 3: FullTokenizer 完整流程")
    print("█" * 80)

    show_step("原始输入", text)

    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case=True)

    # 完整分词
    tokens = tokenizer.tokenize(text)
    show_step("FullTokenizer.tokenize() 输出", tokens,
              "BasicTokenizer + WordpieceTokenizer 的组合结果")

    # Token 转 ID
    ids = tokenizer.convert_tokens_to_ids(tokens)
    show_step("convert_tokens_to_ids()", ids,
              "将 token 转换为词表中的 ID")

    # 显示对应关系
    print(f"\n{'='*80}")
    print("Token → ID 对应关系:")
    print(f"{'='*80}")
    for token, id_ in zip(tokens, ids):
        print(f"  '{token:15s}' → {id_:6d}")

    # ID 转回 Token
    recovered_tokens = tokenizer.convert_ids_to_tokens(ids)
    show_step("convert_ids_to_tokens()", recovered_tokens,
              "将 ID 转换回 token（验证可逆性）")

    print(f"\n验证: {tokens == recovered_tokens}")

    return tokens, ids


def main():
    """主函数"""
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "Tokenization Pipeline 详解" + " " * 31 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80)

    # 测试文本
    test_texts = [
        "latter我喜欢       机器学习和deep learning",
    ]

    vocab_file = "../albert_config/vocab.txt"

    for idx, text in enumerate(test_texts):
        print(f"\n\n{'#'*80}")
        print(f"# 示例 {idx + 1}")
        print(f"{'#'*80}")

        # Part 1: BasicTokenizer 详细流程
        basic_tokens = demo_basic_tokenizer_step_by_step(text)

        # Part 2: WordpieceTokenizer 详细流程
        wordpiece_tokens = demo_wordpiece_tokenizer(basic_tokens, vocab_file)

        # Part 3: 验证完整流程
        final_tokens, final_ids = demo_full_pipeline(text, vocab_file)

        # 验证结果一致性
        print(f"\n{'='*80}")
        print("【验证】分步执行 vs 直接调用 FullTokenizer:")
        print(f"{'='*80}")
        print(f"分步执行结果: {wordpiece_tokens}")
        print(f"直接调用结果: {final_tokens}")
        print(f"结果一致: {wordpiece_tokens == final_tokens}")

        if idx < len(test_texts) - 1:
            input("\n按 Enter 继续下一个示例...")


if __name__ == "__main__":
    main()

