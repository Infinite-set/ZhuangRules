import random
import json
import argparse

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}

def shuffle_dict(d):
    # 获取字典的键值对列表
    items = list(d.items())
    
    # 打乱键值对
    random.shuffle(items)
    
    # 生成一个新的字典，按打乱后的顺序
    shuffled_dict = dict(items)
    
    return shuffled_dict


def construct_prompt_zh2za(data_item, args):

    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，请根据给出的信息将汉语短语或句子翻译成壮语。你的回答应该只包含翻译结果，不要包含任何其他额外信息。\n\n"
    prompt += "# 以下是一条关于壮语的语法规则：\n" + data_item['description'] + "\n\n"

    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇和语法，将汉语短语或句子翻译成壮语。\n\n"
        for i in range(min(len(data_item['examples']), args.num_parallel_sent)):
            example_item = data_item['examples'][i]
            prompt += f"## 请将下面的汉语短语或句子翻译成壮语：{example_item['zh']}\n"
            prompt += "## 在上面的短语或句子中，"
            for za_word, zh_word in example_item['related_words'].items():
                prompt += f"汉语词语“{zh_word}”在壮语中的翻译是“{za_word}”；"
            prompt += f"## 所以，该汉语短语或句子完整的壮语翻译是：{example_item['za']}\n\n"
    
    # prompt最后是需要翻译的短语或句子
    prompt += f"## 请将下面的汉语短语或句子翻译成壮语：{data_item['zh']}\n"
    prompt += "## 在上面的短语或句子中，"
    for za_word, zh_word in data_item['related_words'].items():
        prompt += f"汉语词语“{zh_word}”在壮语中的翻译是“{za_word}”；"
    prompt += f"\n## 所以，该汉语短语或句子完整的壮语翻译是："

    return prompt

def construct_prompt_za2zh(data_item, args):

    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，请根据给出的信息将壮语短语或句子翻译成汉语。你的回答应该只包含翻译结果，不要包含任何其他额外信息。\n\n"
    prompt += "# 以下是一条关于壮语的语法规则：\n" + data_item['description'] + "\n\n"

    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇和语法，将壮语短语或句子翻译成汉语。\n\n"
        for i in range(min(len(data_item['examples']), args.num_parallel_sent)):
            example_item = data_item['examples'][i]
            prompt += f"## 请将下面的壮语短语或句子翻译成汉语：{example_item['za']}\n"
            prompt += "## 在上面的短语或句子中，"
            for za_word, zh_word in example_item['related_words'].items():
                prompt += f"壮语词语“{za_word}”在汉语中的翻译是“{zh_word}”；"
            prompt += f"## 所以，该壮语短语或句子完整的汉语翻译是：{example_item['zh']}\n\n"
    
    # prompt最后是需要翻译的短语或句子
    prompt += f"## 请将下面的壮语短语或句子翻译成汉语：{data_item['za']}\n"
    prompt += "## 在上面的短语或句子中，"
    for za_word, zh_word in data_item['related_words'].items():
        prompt += f"壮语词语“{za_word}”在汉语中的翻译是“{zh_word}”；"
    prompt += f"\n## 所以，该壮语短语或句子完整的汉语翻译是："

    return prompt
    

def construct_prompt_eng2kgv(data_item, args):

    prompt = "# Kalamang is a low-resource language. You are a linguist. Please translate the given English sentence into Kalamang. Your answer should only contain the translation result, without any additional information.\n\n"
    prompt += "# The following is a grammar rule for Kalamang:\n" + data_item['description'] + "\n\n"
    
    # The prompt ends with the phrase or sentence to be translated
    prompt += f"## Please translate the following English sentence into Kalamang based on the given grammar rule: {data_item['eng']}\n"
    prompt += "## In the above phrase or sentence, "
    # random shuffle the related words
    data_item['related_words'] = shuffle_dict(data_item['related_words'])
    for kgv_word, eng_word in data_item['related_words'].items():
        prompt += f"The English word '{eng_word}' translates to '{kgv_word}' in Kalamang; "
    prompt += f"\n## Therefore, the complete Kalamang translation of the English phrase or sentence is:"

    return prompt

def construct_prompt_kgv2eng(data_item, args):

    prompt = "# Kalamang is a low-resource language. You are a linguist. Please translate the given Kalamang sentence into English. Your answer should only contain the translation result, without any additional information.\n\n"
    prompt += "# The following is a grammar rule for Kalamang:\n" + data_item['description'] + "\n\n"

    # The prompt ends with the phrase or sentence to be translated
    prompt += f"## Please translate the following Kalamang sentence into English based on the given grammar rule: {data_item['kgv']}\n"
    prompt += "## In the above phrase or sentence, "
    # random shuffle the related words
    data_item['related_words'] = shuffle_dict(data_item['related_words'])
    for kgv_word, eng_word in data_item['related_words'].items():
        prompt += f"The Kalamang word '{kgv_word}' translates to '{eng_word}' in English; "
    prompt += f"\n## Therefore, the complete English translation of the Kalamang phrase or sentence is:"

    return prompt

def construct_prompt_eng2kgv_no_grammar(data_item, args):

    prompt = "# Kalamang is a low-resource language. You are a linguist. Please translate the given English sentence into Kalamang. Your answer should only contain the translation result, without any additional information.\n\n"
    
    # The prompt ends with the phrase or sentence to be translated
    prompt += f"## Please translate the following English sentence into Kalamang: {data_item['eng']}\n"
    prompt += "## In the above phrase or sentence, "
    # random shuffle the related words
    data_item['related_words'] = shuffle_dict(data_item['related_words'])
    for kgv_word, eng_word in data_item['related_words'].items():
        prompt += f"The English word '{eng_word}' translates to '{kgv_word}' in Kalamang; "
    prompt += f"\n## Therefore, the complete Kalamang translation of the English phrase or sentence is:"

    return prompt

def construct_prompt_kgv2eng_no_grammar(data_item, args):

    prompt = "# Kalamang is a low-resource language. You are a linguist. Please translate the given Kalamang sentence into English. Your answer should only contain the translation result, without any additional information.\n\n"

    # The prompt ends with the phrase or sentence to be translated
    prompt += f"## Please translate the following Kalamang sentence into English: {data_item['kgv']}\n"
    prompt += "## In the above phrase or sentence, "
    # random shuffle the related words
    data_item['related_words'] = shuffle_dict(data_item['related_words'])
    for kgv_word, eng_word in data_item['related_words'].items():
        prompt += f"The Kalamang word '{kgv_word}' translates to '{eng_word}' in English; "
    prompt += f"\n## Therefore, the complete English translation of the Kalamang phrase or sentence is:"

    return prompt



def construct_prompt_zh2za_no_grammar(data_item, args):

    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，请根据给出的信息将汉语短语或句子翻译成壮语。你的回答应该只包含翻译结果，不要包含任何其他额外信息。\n\n"

    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇，将汉语短语或句子翻译成壮语。\n\n"
        for i in range(min(len(data_item['examples']), args.num_parallel_sent)):
            example_item = data_item['examples'][i]
            prompt += f"## 请将下面的汉语短语或句子翻译成壮语：{example_item['zh']}\n"
            prompt += "## 在上面的短语或句子中，"
            for za_word, zh_word in example_item['related_words'].items():
                prompt += f"汉语词语“{zh_word}”在壮语中的翻译是“{za_word}”；"
            prompt += f"## 所以，该汉语短语或句子完整的壮语翻译是：{example_item['za']}\n\n"
    
    # prompt最后是需要翻译的短语或句子
    prompt += f"## 请将下面的汉语短语或句子翻译成壮语：{data_item['zh']}\n"
    prompt += "## 在上面的短语或句子中，"
    for za_word, zh_word in data_item['related_words'].items():
        prompt += f"汉语词语“{zh_word}”在壮语中的翻译是“{za_word}”；"
    prompt += f"\n## 请根据以上词汇表完成该句子的翻译。如果需要翻译的句子中包含了以上词汇表中不存在的词汇，你应该忽略这些词汇，只使用词汇表中存在的词汇的翻译作为该句子的翻译。"
    prompt += f"\n## 所以，该汉语短语或句子完整的壮语翻译是："

    return prompt

def construct_prompt_za2zh_no_grammar(data_item, args):

    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，请根据给出的信息将壮语短语或句子翻译成汉语。你的回答应该只包含翻译结果，不要包含任何其他额外信息。\n\n"

    if args.num_parallel_sent > 0:
        prompt += "# 请仿照样例，参考给出的词汇，将壮语短语或句子翻译成汉语。\n\n"
        for i in range(min(len(data_item['examples']), args.num_parallel_sent)):
            example_item = data_item['examples'][i]
            prompt += f"## 请将下面的壮语短语或句子翻译成汉语：{example_item['za']}\n"
            prompt += "## 在上面的短语或句子中，"
            for za_word, zh_word in example_item['related_words'].items():
                prompt += f"壮语词语“{za_word}”在汉语中的翻译是“{zh_word}”；"
            prompt += f"## 所以，该壮语短语或句子完整的汉语翻译是：{example_item['zh']}\n\n"
    
    # prompt最后是需要翻译的短语或句子
    prompt += f"## 请将下面的壮语短语或句子翻译成汉语：{data_item['za']}\n"
    prompt += "## 在上面的短语或句子中，"
    for za_word, zh_word in data_item['related_words'].items():
        prompt += f"壮语词语“{za_word}”在汉语中的翻译是“{zh_word}”；"
    prompt += f"\n## 请根据以上词汇表完成该句子的翻译。如果需要翻译的句子中包含了以上词汇表中不存在的词汇，你应该忽略这些词汇，只使用词汇表中存在的词汇的翻译作为该句子的翻译。"
    prompt += f"\n## 所以，该壮语短语或句子完整的汉语翻译是："

    return prompt


def construct_prompt_za2zh_igt(data_item, args):
    all_igt_dict = json.load(open(args.igt_path, 'r'))

    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，请根据给出的信息将汉语短语或句子翻译成壮语。你的回答应该只包含翻译结果，不要包含任何其他额外信息。\n\n"
    prompt += "# 以下是一些有关壮语语法的例句，以及它们的IGT(Interlinear Glossed Text)，可以帮助你完成翻译：\n\n"

    if args.num_parallel_sent > 0:
        for i in range(min(len(data_item['examples']), args.num_parallel_sent)):
            prompt += f"## 例句{i+1}：\n"
            prompt += "字典为："
            za2zh_dict = {k: v for k, v in data_item['examples'][i]['related_words'].items()}
            prompt += json.dumps(za2zh_dict, ensure_ascii=False)
            prompt += f"\n壮语：{data_item['examples'][i]['za']}\n"
            prompt += f"IGT：{all_igt_dict[data_item['examples'][i]['test_instance_id']]['igt']}\n"
            prompt += f"汉语：{data_item['examples'][i]['zh']}\n\n"
    
    # prompt最后是需要翻译的短语或句子
    prompt += f"## 请首先写出下面的壮语短语或句子的IGT，然后将其翻译成汉语：\n"
    prompt += f"字典为："
    za2zh_dict = {k: v for k, v in data_item['related_words'].items()}
    prompt += json.dumps(za2zh_dict, ensure_ascii=False)
    prompt += f"\n壮语：{data_item['za']}\n"
    prompt += f"\n## 所以，该壮语短语或句子的IGT和汉语翻译分别是："

    return prompt

def construct_prompt_za2zh_igt_grammar(data_item, args):
    all_igt_dict = json.load(open(args.igt_path, 'r'))

    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，请根据给出的信息将汉语短语或句子翻译成壮语。你的回答应该只包含翻译结果，不要包含任何其他额外信息。\n\n"
    prompt += "# 以下是一条关于壮语的语法规则：\n" + data_item['description'] + "\n\n"
    prompt += "# 以下为该规则的一些例句及其IGT(Interlinear Glossed Text)，可以帮助你完成翻译：\n\n"

    if args.num_parallel_sent > 0:
        for i in range(min(len(data_item['examples']), args.num_parallel_sent)):
            prompt += f"## 例句{i+1}：\n"
            prompt += "字典为："
            za2zh_dict = {k: v for k, v in data_item['examples'][i]['related_words'].items()}
            prompt += json.dumps(za2zh_dict, ensure_ascii=False)
            prompt += f"\n壮语：{data_item['examples'][i]['za']}\n"
            prompt += f"IGT：{all_igt_dict[data_item['examples'][i]['test_instance_id']]['igt']}\n"
            prompt += f"汉语：{data_item['examples'][i]['zh']}\n\n"
    
    # prompt最后是需要翻译的短语或句子
    prompt += f"## 请首先写出下面的壮语短语或句子的IGT，然后将其翻译成汉语：\n"
    prompt += f"字典为："
    za2zh_dict = {k: v for k, v in data_item['related_words'].items()}
    prompt += json.dumps(za2zh_dict, ensure_ascii=False)
    prompt += f"\n壮语：{data_item['za']}\n"
    prompt += f"\n## 所以，该壮语短语或句子的IGT和汉语翻译分别是："

    return prompt

    