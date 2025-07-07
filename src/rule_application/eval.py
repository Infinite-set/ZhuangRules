import json
import numpy as np
import argparse
import re

from sacrebleu.metrics import BLEU, CHRF

def check_english_char(char):
    return 'a' <= char <= 'z' or 'A' <= char <= 'Z'

def check_chinese_char(char):
    # 检查是否为汉字
    return '\u4e00' <= char <= '\u9fff'

def post_process(text, lang):
    charac = ['。', '.', '？', '?', '！', '!', '：', ':', '；', ';', '`', '\n', '*', '>', '=', '“', '”', '"', '‘', '’', '《', '》', '(', ')', '（', '）', '【', '】', '[', ']', '「', '」', '『', '』', '〈', '〉', '〔', '〕', '〖', '〗', '〘', '〙', '〚', '〛', '〝', '〞', '〟', '〾', '〿', '–', '—', '―', '‖', '‗', '‘', '’', '‚', '‛', '“', '”', '„', '‟', '‹', '›', '‽', '⁇', '⁈', '⁉', '⁊', '⁋', '⁌', '⁍', '⁎', '⁏', '⁐', '⁑', '⁒', '⁓', '⁔', '⁕', '⁖', '⁗', '⁘', '⁙', '⁚', '⁛', '⁜', '⁝', '⁞']
    # 从右往左，在出现过英文字母后的第一个标点类字符
    for i in range(len(text) - 1, -1, -1):
        if lang == 'za':
            check = check_english_char(text[i])
        elif lang == 'zh':
            check = check_chinese_char(text[i])
        else:
            raise ValueError("lang must be 'za' or 'zh'")
        if check:
            for j in range(i, -1, -1):
                if text[j] in charac:
                    return text[j+1:i+1]
                
def split_igt(text):
    text = text.split('\n')
    if len(text) == 1:
        return text[0]
    else:
        text = text[-1]
        for i in range(len(text) - 1, -1, -1):
            if text[i] == ":" or text[i] == "：":
                return text[i+1:]
        return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='pred.jsonl')
    parser.add_argument('--lang', type=str, default='zh')
    parser.add_argument('--leveled', action='store_true')
    parser.add_argument('--detailed', action='store_true')
    parser.add_argument('--wals', action='store_true')
    parser.add_argument('--wals_path', type=str, default='data/wals/wals_annotations.json')
    args = parser.parse_args()

    if args.lang == 'zh':
        chrfpp = CHRF(word_order=2, lowercase=True)
        chrf = CHRF(word_order=0, lowercase=True)
        scarebleu = BLEU(lowercase=True, tokenize='zh', effective_order=True)
    elif args.lang in ['za', 'eng', 'kgv']:
        chrfpp = CHRF(word_order=2, lowercase=True)
        chrf = CHRF(word_order=0, lowercase=True)
        scarebleu = BLEU(lowercase=True, effective_order=True)
    

    if args.detailed:
        all_data = json.load(open(args.metadata_path, 'r'))

    if args.wals:
        wals_data = json.load(open(args.wals_path, 'r'))
        



    # load data
    # 根据id字段，各个rule内部求平均，再在所有rule上求平均
    output = {}
    total_num = 0
    with open(args.output_path, "r") as f:
        for line in f:
            line = json.loads(line)
            rule_id = line['id'].split('-')[1]
            if rule_id not in output:
                output[rule_id] = [line]
            else:
                output[rule_id].append(line)
            total_num += 1

    add_item = []

    print("total_num:", total_num)
    metrics = []    
    if args.detailed:
        detailed_results = {}
        difficulty_results = {}
    if args.wals:
        wals_results = {}

    for rule_id, items in output.items():
        refs = []
        preds = []
        for item in items:
            if args.lang == 'za':
                item['gold'] = item['gold'].replace("’", "'").replace("‘", "'").replace("。", ".")
                item['pred'] = item['pred'].replace("’", "'").replace("‘", "'").replace("。", ".")

                if '# target_sentence = ' in item['pred']:
                    item['pred'] = item['pred'].split('# target_sentence = ')[1].split('\n')[0]
                elif '# final_results = ' in item['pred']:
                    item['pred'] = item['pred'].split('# final_results = ')[1].split('\n')[0]
                elif '# final_translation = ' in item['pred']:
                    item['pred'] = item['pred'].split('# final_translation = ')[1].split('\n')[0]
                else:
                    pass

                post_processed_pred = post_process(item['pred'], args.lang)
                if post_processed_pred is not None:
                    item['pred'] = post_processed_pred.strip()

            if args.lang == 'zh' and 'multi_rules' in args.output_path:
                post_processed_pred = post_process(item['pred'], args.lang)
                if post_processed_pred is not None:
                    item['pred'] = post_processed_pred.strip()

            if 'llama' in args.output_path:
                item['pred'] = item['pred'].split('#')[0].strip()

            if args.lang == 'kgv':
                item['pred'] = item['pred'].replace("\"", "").strip()
                item['pred'] = item['pred'].replace("-", " ").strip()

            if 'igt' in args.output_path:
                item['pred'] = split_igt(item['pred'])
            
            if item['gold'].endswith('.') or item['gold'].endswith('。'):
                item['gold'] = item['gold'][:-1]
            if item['pred'].endswith('.') or item['pred'].endswith('。'):
                item['pred'] = item['pred'][:-1]

            refs.append(item['gold'])
            preds.append(item['pred'])

        refs = [refs]
        result = {
            'rule_id': rule_id,
            'sacrebleu': scarebleu.corpus_score(preds, refs).score,
            'chrf++': chrfpp.corpus_score(preds, refs).score,
            'chrf': chrf.corpus_score(preds, refs).score
        }
        metrics.append(result)

        if args.detailed:
            this_dt = None
            for dt in all_data:
                if dt['rule_id'].split('-')[1] == rule_id:
                    this_dt = dt
                    break
            assert this_dt is not None
            for item in this_dt["operations"]:
                if item not in detailed_results:
                    detailed_results[item] = [result]
                else:
                    detailed_results[item].append(result)
            if this_dt["difficulty"] not in difficulty_results:
                difficulty_results[this_dt["difficulty"]] = [result]
            else:
                difficulty_results[this_dt["difficulty"]].append(result)

        if args.wals:
            this_dt = None
            for dt in wals_data:
                if dt['rule_id'].split('-')[1] == rule_id:
                    this_dt = dt
                    break
            assert this_dt is not None
            if this_dt["wals"]["area"] not in wals_results:
                wals_results[this_dt["wals"]["area"]] = [result]
            else:
                wals_results[this_dt["wals"]["area"]].append(result)

    metrics_meaned = {}
    metrics_meaned['sacrebleu'] = np.mean([item['sacrebleu'] for item in metrics])
    metrics_meaned['chrf++'] = np.mean([item['chrf++'] for item in metrics])
    metrics_meaned['chrf'] = np.mean([item['chrf'] for item in metrics])

    for key in ['sacrebleu', 'chrf++', 'chrf']:
        metrics_meaned[key] = np.around(metrics_meaned[key], decimals=4)
    print("overall:")
    print(metrics_meaned)

    if args.detailed:
        for item, results in detailed_results.items():
            result = {
                'sacrebleu': np.mean([item['sacrebleu'] for item in results]),
                'chrf++': np.mean([item['chrf++'] for item in results]),
                'chrf': np.mean([item['chrf'] for item in results])
            }
            for key in ['sacrebleu', 'chrf++', 'chrf']:
                result[key] = np.around(result[key], decimals=4)
            print(item + ":\t\t" + json.dumps(result))
        for item, results in difficulty_results.items():
            result = {
                'sacrebleu': np.mean([item['sacrebleu'] for item in results]),
                'chrf++': np.mean([item['chrf++'] for item in results]),
                'chrf': np.mean([item['chrf'] for item in results])
            }
            for key in ['sacrebleu', 'chrf++', 'chrf']:
                result[key] = np.around(result[key], decimals=4)
            print(item + ":\t\t" + json.dumps(result))

    if args.wals:
        for item, results in wals_results.items():
            result = {
                'sacrebleu': np.mean([item['sacrebleu'] for item in results]),
                'chrf++': np.mean([item['chrf++'] for item in results]),
                'chrf': np.mean([item['chrf'] for item in results])
            }
            for key in ['sacrebleu', 'chrf++', 'chrf']:
                result[key] = np.around(result[key], decimals=4)
            print(item + ":\t\t" + json.dumps(result))