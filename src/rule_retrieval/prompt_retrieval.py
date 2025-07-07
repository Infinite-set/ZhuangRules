import random
import json
import argparse

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}


def construct_prompt_zh2za_selection(data_item, args):

    all_data = json.load(open(args.grammar_path, 'r'))
    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，以下是一部壮语语法书。给你一条需要翻译的汉语短语或句子，请根据语法书内容，选出适用于翻译该短语或句子的语法规则。你需要挑选出五条你认为最可能适用的语法规则，你的回答应该只包含这五条语法规则的描述，并且按照适用的可能性从大到小进行排列，不要包含任何其他额外信息。\n\n"
    for gram_idx, data in enumerate(all_data):
        prompt += f"## 语法规则 {gram_idx+1}：\n"
        prompt += f"### 语法规则描述：{data['description']}\n"
        prompt += "\n"
    # prompt最后是需要翻译的短语或句子
    prompt += f"# 现在，请挑选出最可能适用于翻译下面汉语短语或句子的五条语法规则：{data_item['zh']}\n"
    prompt += "## 字典为："
    zh2za_dict = {v: k for k, v in data_item['related_words'].items()}
    prompt += json.dumps(zh2za_dict, ensure_ascii=False)
    prompt += f"\n## 所以，翻译该汉语短语或句子需要的壮语语法规则是："
    return prompt

def construct_prompt_za2zh_selection(data_item, args):

    all_data = json.load(open(args.grammar_path, 'r'))
    prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，以下是一部壮语语法书。给你一条需要翻译的壮语短语或句子，请根据语法书内容，选出适用于翻译该短语或句子的语法规则。你需要挑选出五条你认为最可能适用的语法规则，你的回答应该只包含这五条语法规则的描述，并且按照适用的可能性从大到小进行排列，不要包含任何其他额外信息。\n\n"
    for gram_idx, data in enumerate(all_data):
        prompt += f"## 语法规则 {gram_idx+1}：\n"
        prompt += f"### 语法规则描述：{data['description']}\n"
        prompt += "\n"
    # prompt最后是需要翻译的短语或句子
    prompt += f"# 现在，请挑选出最可能适用于翻译下面壮语短语或句子的五条语法规则：{data_item['za']}\n"
    prompt += "## 字典为："
    za2zh_dict = {k: v for k, v in data_item['related_words'].items()}
    prompt += json.dumps(za2zh_dict, ensure_ascii=False)
    prompt += f"\n## 所以，翻译该壮语短语或句子需要的壮语语法规则是："
    return prompt



def construct_prompt_zh2za_iterative_rules(data_items, args, rule_idx):
    if args.use_code:
        all_data = json.load(open(args.code_grammar_path, 'r'))

        prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，以下是一条壮语语法规则的相关信息。给你一些需要翻译为壮语的汉语短语或句子，请根据该语法规则的内容，逐一检查翻译过程中是否需要使用该规则。你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），不包含任何其他额外信息。\n\n"

        prompt += f"## 语法规则：\n"
        prompt += f"### 语法规则描述：{all_data[rule_idx]['description']}\n"
        prompt += f"### 检查汉语句子是否需要使用该语法规则进行翻译的伪代码如下：{all_data[rule_idx]['code']}\n\n"
        prompt += "## 请逐一检查下面的汉语短语或句子，判断是否需要使用该语法规则进行翻译：\n"

        for data_idx, data_item in enumerate(data_items):
            prompt += f"### 汉语短语或句子 {data_idx+1}：{data_item['zh']}\n\n"
            prompt += "### 在上面的短语或句子中，"
            for za_word, zh_word in data_item['related_words'].items():
                prompt += f"汉语词语“{zh_word}”在壮语中的翻译是“{za_word}”；"
            prompt += f"\n\n"
            
        prompt += "请逐一检查上面的汉语短语或句子，对每一个汉语短语或句子，你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），且不包含任何其他额外信息。"

        return prompt

    else:
        all_data = json.load(open(args.grammar_path, 'r'))

        prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，以下是一条壮语语法规则的相关信息。给你一些需要翻译为壮语的汉语短语或句子，请根据该语法规则的内容，逐一检查翻译过程中是否需要使用该规则。你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），不包含任何其他额外信息。\n\n"

        prompt += f"## 语法规则：\n"
        prompt += f"### 语法规则描述：{all_data[rule_idx]['description']}\n"
        prompt += "## 请逐一检查下面的汉语短语或句子，判断是否需要使用该语法规则进行翻译：\n\n"

        for data_idx, data_item in enumerate(data_items):
            prompt += f"### 汉语短语或句子 {data_idx+1}：{data_item['zh']}\n"
            prompt += "### 在上面的短语或句子中，"
            for za_word, zh_word in data_item['related_words'].items():
                prompt += f"汉语词语“{zh_word}”在壮语中的翻译是“{za_word}”；"
            prompt += f"\n\n"

        prompt += "请逐一检查上面的汉语短语或句子，对每一个汉语短语或句子，你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），且不包含任何其他额外信息。"

        return prompt
    

def construct_prompt_za2zh_iterative_rules(data_items, args, rule_idx):
    if args.use_code:
        all_data = json.load(open(args.code_grammar_path, 'r'))

        prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，以下是一条壮语语法规则的相关信息。给你一些需要翻译为汉语的壮语短语或句子，请根据该语法规则的内容，逐一检查翻译过程中是否需要使用该规则。你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），不包含任何其他额外信息。\n\n"

        prompt += f"## 语法规则：\n"
        prompt += f"### 语法规则描述：{all_data[rule_idx]['description']}\n"
        prompt += f"### 检查壮语句子是否需要使用该语法规则进行翻译的伪代码如下：{all_data[rule_idx]['code']}\n\n"
        prompt += "## 请逐一检查下面的壮语短语或句子，判断是否需要使用该语法规则进行翻译：\n"

        for data_idx, data_item in enumerate(data_items):
            prompt += f"### 壮语短语或句子 {data_idx+1}：{data_item['za']}\n\n"
            prompt += "### 在上面的短语或句子中，"
            for za_word, zh_word in data_item['related_words'].items():
                prompt += f"壮语词语“{za_word}”在汉语中的翻译是“{zh_word}”；"
            prompt += f"\n\n"
            
        prompt += "请逐一检查上面的壮语短语或句子，对每一个壮语短语或句子，你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），且不包含任何其他额外信息。"

        return prompt

    else:
        all_data = json.load(open(args.grammar_path, 'r'))

        prompt = "# 壮语是中国的一门少数民族语言。你是一名语言学家，以下是一条壮语语法规则的相关信息。给你一些需要翻译为汉语的壮语短语或句子，请根据该语法规则的内容，逐一检查翻译过程中是否需要使用该规则。你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），不包含任何其他额外信息。\n\n"

        prompt += f"## 语法规则：\n"
        prompt += f"### 语法规则描述：{all_data[rule_idx]['description']}\n"
        prompt += "## 请逐一检查下面的壮语短语或句子，判断是否需要使用该语法规则进行翻译：\n\n"

        for data_idx, data_item in enumerate(data_items):
            prompt += f"### 壮语短语或句子 {data_idx+1}：{data_item['za']}\n"
            prompt += "### 在上面的短语或句子中，"
            for za_word, zh_word in data_item['related_words'].items():
                prompt += f"壮语词语“{za_word}”在汉语中的翻译是“{zh_word}”；"
            prompt += f"\n\n"

        prompt += "请逐一检查上面的壮语短语或句子，对每一个壮语短语或句子，你的回答中只应包含是否需要使用该规则的判断（“是”或“否”），且不包含任何其他额外信息。"

        return prompt