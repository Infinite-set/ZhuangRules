# chinese to zhuang
cd ./src/application

python3 main.py \
--src_lang zh \
--tgt_lang za \
--test_data_path data/zhuangrules_v0_for_test.json \
--model_name /your/model/name \
--model_path /path/to/your/model \
--prompt_type zh2za_code \
--num_parallel_sent 2 \
--output_path /path/to/save/your/output.jsonl \
--max_new_tokens 1000