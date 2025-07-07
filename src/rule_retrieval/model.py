import json
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name, model_path, n_gpu=1):
    print("loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    print("loaded!")
    return llm, tokenizer
    
    

def get_pred(llm, sampling_params, prompt):
    # print("prompt:", prompt)
    outputs = llm.generate(prompt, sampling_params)
    result = outputs[0].outputs[0].text.strip().split('\n')[0]
    result = result.split("<|endoftext|>")[0].strip()
    result = result.split("<|im_end|>")[0].strip()
    
    # print("result:", result)
    return result

def get_pred_no_vllm(llm, tokenizer, prompt, args):
    if args.model_name == "qwen":
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("prompt:", prompt)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = len(inputs['input_ids'][0])
    # print("input_len", input_len)
    inputs = inputs.to('cuda')
    preds = llm.generate(
        **inputs,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )
    pred = tokenizer.decode(preds[0][input_len:], skip_special_tokens=True)
    output = pred
    if args.model_name != "qwen":
        result = output.strip().split('\n\n')[0]
    else:
        result = output.strip()
    return result


if __name__ == '__main__':
    pass