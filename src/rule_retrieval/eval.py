import json
import numpy as np
import argparse
import re
import jieba

from rank_bm25 import BM25Okapi
from sacrebleu.metrics import BLEU, CHRF


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='pred.jsonl')
    parser.add_argument('--lang', type=str, default='zh')
    args = parser.parse_args()
    

    # load data
    output = {}
    with open(args.output_path, "r") as f:
        for line in f:
            line = json.loads(line)
            rule_id = line['id'].split('-')[1]
            if rule_id not in output:
                output[rule_id] = [line]
            else:
                output[rule_id].append(line)

    dictionary_path = './data/dictionarioes.json'
    dictionary = json.load(open(dictionary_path, "r"))

    tokenized_corpus = [list(jieba.cut(rule)) for rule in dictionary]
    bm25 = BM25Okapi(tokenized_corpus)

    gold_data_path = './data/zhuangrules_v0_for_test.json'
    gold_data = json.load(open(gold_data_path, "r"))

    acc = []

    error_rule_type = {}
    total_rule_type = {}

    for rule_id, items in output.items():

        rule_type = []
        original_data = json.load(open('./data/zhuangrules_v0.json', "r"))
        for item in original_data:
            if item['rule_id'] == ("rule-" + rule_id):
                rule_type = item['operations']
                break


        acc_rule = []
        for item in items:

            for typ in rule_type:
                if typ not in total_rule_type:
                    total_rule_type[typ] = 1
                else:
                    total_rule_type[typ] += 1

            query = list(jieba.cut(item['pred']))
            result = bm25.get_top_n(query, dictionary, n=1)

            for gold_item in gold_data:
                if gold_item['id'] == item['id']:
                    gold_rule = gold_item["description"]
                    break
        

            if gold_rule in result:
                acc_rule.append(1)
            else:
                acc_rule.append(0)
                for typ in rule_type:
                    if typ not in error_rule_type:
                        error_rule_type[typ] = 1
                    else:
                        error_rule_type[typ] += 1

        acc.append(np.mean(acc_rule))

    print("acc:", np.mean(acc))
    print("error_rule_type:", error_rule_type)
    print("total_rule_type:", total_rule_type)