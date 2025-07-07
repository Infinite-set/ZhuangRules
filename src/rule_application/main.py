import json
import argparse
import os
import random

from tqdm import tqdm
import numpy as np
import torch

from model import load_model, get_pred_no_vllm # vllm is also supported in model.py
from prompt import *
from prompt_code import *

if __name__ == '__main__':

    random.seed(42)

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--chat_mode', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)

    # config for generation
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.05)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=100)

    # data
    parser.add_argument('--test_data_path', type=str, default='data')

    # config for prompt
    parser.add_argument('--src_lang', type=str, default='zh')
    parser.add_argument('--tgt_lang', type=str, default='za')
    parser.add_argument('--prompt_type', type=str, default='za2zh')
    parser.add_argument('--num_parallel_sent', type=int, default=5)

    # output path
    parser.add_argument('--output_path', type=str, default=None)

    parser.add_argument('--code_grammar_path', type=str, default='data/code_grammars/code_grammars_zh2za.json', help="Path to the code grammar rules if needed.")
    parser.add_argument('--igt_path', type=str, default='data/igt/igt_gpt-4o-2024-11-20_output_dict.json', help="Path to the IGT data if needed.")


    args = parser.parse_args()

    # load test data
    if args.test_data_path.endswith('.jsonl'):
        test_data = [json.loads(line) for line in open(args.test_data_path, "r")]
    else:
        test_data = json.load(open(args.test_data_path, "r"))

    # load model
    # llm, tokenizer = None, None # 测试时不加载模型
    llm, tokenizer = load_model(args.model_name, args.model_path, args.n_gpu)

    # construct prompt 
    prompt_func = None

    prompt_type_to_prompt_func = {
        'za2zh': construct_prompt_za2zh, # provided: grammar rule + num_parallel_sent * parallel examples (set num_parallel_sent=0 for grammar rule only)
        'zh2za': construct_prompt_zh2za,
        'za2zh_no_grammar': construct_prompt_za2zh_no_grammar,  # provided: num_parallel_sent * parallel examples
        'zh2za_no_grammar': construct_prompt_zh2za_no_grammar,
        'za2zh_code': construct_prompt_za2zh_code,  # provided: code grammar rule + num_parallel_sent * parallel examples (set num_parallel_sent=0 for code grammar rule only)
        'zh2za_code': construct_prompt_zh2za_code,
        'za2zh_igt': construct_prompt_za2zh_igt,    # provided: num_parallel_sent * parallel examples with igt
        'za2zh_igt_grammar': construct_prompt_za2zh_igt_grammar,    # provided: grammar rule + num_parallel_sent * parallel examples with igt
        'za2zh_igt_code': construct_prompt_za2zh_igt_code,  # provided: code grammar rule + num_parallel_sent * parallel examples with igt
        'kgv2eng': construct_prompt_kgv2eng,
        'eng2kgv': construct_prompt_eng2kgv,
        'kgv2eng_code': construct_prompt_kgv2eng_code,
        'eng2kgv_code': construct_prompt_eng2kgv_code,
        'eng2kgv_no_grammar': construct_prompt_eng2kgv_no_grammar,
        'kgv2eng_no_grammar': construct_prompt_kgv2eng_no_grammar,
    }

    if args.prompt_type not in prompt_type_to_prompt_func:
        raise NotImplementedError("Unsupported prompt type!")
    else:
        prompt_func = prompt_type_to_prompt_func[args.prompt_type]


    # chat mode
    if args.chat_mode:
        if 'qwen' in args.model_name and 'chat' in args.model_name:
            chat_template = model_to_chat_template['qwen']


    # output path
    if args.output_path == None:
        args.output_path = f"../output/{args.model_name}_{args.prompt_type}_parallel{args.num_parallel_sent}.jsonl"
    
    while os.path.exists(args.output_path):
        args.output_path = args.output_path + ".new.jsonl"
    print("output_path:", args.output_path)
    fout = open(args.output_path, "w")

    # do test
    for item in tqdm(test_data):

        src_sentence = None
        if args.src_lang in item:
            src_sentence = item[args.src_lang]
        else:
            src_sentence = item['query']

        # construct prompt
        prompt = prompt_func(item, args)
        # print(prompt)

        # special treatment for chat mode
        if args.chat_mode:
            prompt = chat_template.format(prompt=prompt)

        # generate
        pred = get_pred_no_vllm(llm, tokenizer, prompt, args)
        

        print("input:", src_sentence)
        print("gold:", item[args.tgt_lang] if args.tgt_lang in item else item["gold"])
        print("pred:", pred.split("##")[0])

        fout.write(json.dumps({"query": src_sentence, "pred": pred, "gold": item[args.tgt_lang] if args.tgt_lang in item else item["gold"], "prompt": prompt, "id": item["id"]}, ensure_ascii=False) + "\n")
        fout.flush()
