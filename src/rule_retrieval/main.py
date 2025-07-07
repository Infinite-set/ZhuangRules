import json
import argparse
import os
import random

from tqdm import tqdm
import numpy as np
import torch

from model import load_model, get_pred_no_vllm
from prompt_retrieval import *

if __name__ == '__main__':

    random.seed(42)

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--chat_mode', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)

    # config for generation
    parser.add_argument('--do_sample', action='store_true')
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

    # output path
    parser.add_argument('--output_path', type=str, default=None)

    parser.add_argument('--use_code', action='store_true', help="Whether to use code in the prompt. If False, only natural language grammar are used.")

    # iterative rules 时，一次inference的句子数
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--grammar_path', type=str, default='data/zhuangrules_v0.json', help="Path to the grammar rules if needed.")
    parser.add_argument('--code_grammar_path', type=str, default='data/code_grammars/code_grammars4retrieval_zh2za.json', help="Path to the code grammar rules if needed.")




    args = parser.parse_args()

    # load test data
    if args.test_data_path.endswith('.jsonl'):
        test_data = [json.loads(line) for line in open(args.test_data_path, "r")]
    else:
        test_data = json.load(open(args.test_data_path, "r"))

    grammar_num = len(json.load(open("./data/zhuang_dictionaries.json", 'r')))
    # assert grammar_num == 109

    # load model
    llm, tokenizer = load_model(args.model_name, args.model_path, args.n_gpu)

    # construct prompt 
    prompt_func = None

    prompt_type_to_prompt_func = {
        'za2zh_iterative_rules': construct_prompt_za2zh_iterative_rules,
        'zh2za_iterative_rules': construct_prompt_zh2za_iterative_rules,
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

    for test_data_idx in tqdm(range(0, len(test_data), args.batch_size)):

        data_items = test_data[test_data_idx: test_data_idx + args.batch_size]

        batch_output = []
        for rule_idx in tqdm(range(grammar_num)):
            # construct prompt
            prompt = prompt_func(data_items, args, rule_idx)

            # special treatment for chat mode
            if args.chat_mode:
                prompt = chat_template.format(prompt=prompt)

            # generate
            pred = get_pred_no_vllm(llm, tokenizer, prompt, args)

            if batch_output == []:
                batch_output = {"indexes": [dt["id"] for dt in data_items], "preds": [pred]}
            else:
                batch_output["preds"].append(pred)

        fout.write(json.dumps(batch_output, ensure_ascii=False) + "\n")
        fout.flush()