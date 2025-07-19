import sys, os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
# from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
#import tiktoken
import torch.multiprocessing as mp
import torch
import math
import random
import numpy as np
from termcolor import colored


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from model_hub import LlamaModel, QwenModel
from datasets import load_dataset
dataset = load_dataset('THUDM/LongBench-v2', split='train')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"], help="Dtype")
    parser.add_argument("--attn_type", type=str, default="RetroInfer",                                                      \
                        choices=["Full_Flash_Attn", "RetroInfer"], help="Attention method")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",                  \
                        choices=["gradientai/Llama-3-8B-Instruct-Gradient-1048k", "Qwen/Qwen2.5-7B-Instruct",               \
                        "Qwen/Qwen2.5-72B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"], help="huggingface model name")
    parser.add_argument("--data_path", type=str, default="", help="Input json file path")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--rag", "-rag", type=int, default=0)
    parser.add_argument("--no_context", "-nc", action='store_true')
    parser.add_argument("--appendix", "-a", type=str, default="")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()
    
    return args


def load_model(model_name, max_len, dtype, device):
    if 'Llama' in model_name:
        llm = LlamaModel(model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    elif 'Qwen' in model_name:
        llm = QwenModel(model_name,
            max_length=max_len,
            dtype=dtype,
            device_map=device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return llm


def generate_config(model_name, context_len, attn_type):
    CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
    MODEL_NAME = model_name.split("/")[-1]+'.json'
    CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
    with open(CONFIG_FILE, "r") as f:
        original_config = json.load(f)
    
    n_clusters = max(int(context_len/16), 1)
    n_segments = max(int(context_len/8192), 1)
    # compute the nearest multiple of (n_segments*32)
    lower = (n_clusters // (n_segments*32)) * (n_segments*32)
    upper = lower + (n_segments*32)
    n_clusters = lower if abs(n_clusters - lower) <= abs(n_clusters - upper) else upper
    nprobe = max(int(n_clusters*0.018), 1)

    if attn_type == 'RetroInfer':
        original_config[attn_type]['n_centroids'] = n_clusters
        original_config[attn_type]['n_segment'] = n_segments
        original_config[attn_type]['nprobe'] = nprobe
        original_config[attn_type]['cache_cluster_num'] = nprobe*3
        original_config[attn_type]['max_compute_cluster_num'] = max(int(n_clusters/4), nprobe)
    
    if attn_type != "Full_Flash_Attn":
        print(original_config[attn_type])
    
    return original_config


def query_llm(data):
    model_name = args.model_name
    batch_size = args.batch_size
    attn_type = args.attn_type
    dtype = torch.float16 if args.dtype=='fp16' else torch.bfloat16
    device = args.device
    data_path = args.data_path

    prompt = []
    groundtruth = []
    for dd in data:
        prompt.append(dd['input'])
        groundtruth.append(dd['outputs'])
    
    # copy to fit batch size
    copy_round = math.ceil(batch_size/len(prompt))
    prompts = []
    groundtruths = []
    for i in range(copy_round):
        prompts.extend(prompt)
        groundtruths.extend(groundtruth)
    prompts = prompts[:batch_size]
    groundtruths = groundtruths[:batch_size]

    # tokenize input data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask

    input_len = input_ids.shape[1]
    gen_len = args.gen_len
    max_len = input_len + gen_len
    print(colored(f"Input length: {input_len}", 'yellow'))

    if data_path == "":
        attn_config = generate_config(model_name, 122880, attn_type)
    else:
        attn_config = generate_config(model_name, input_len, attn_type)

    llm = load_model(model_name, max_len, dtype, device)
    out = llm.generate(attention_type=attn_type,
        inputs_ids = input_ids.to(llm.layers[0].device),
        attention_masks = attention_masks.to(llm.layers[0].device),
        max_new_length=gen_len, attn_config=attn_config)
    
    result = tokenizer.batch_decode(out, skip_special_tokens=True)
    for gt, res in zip(groundtruths, result):
        print(colored(f"Answer: {gt}", 'yellow'))
        print(f"{[res]}")
    return result

    

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_pred(data, args, fout):


    # client = OpenAI(
    #     base_url=URL,
    #     api_key=API_KEY
    # )
    client = None
    for item in tqdm(data):
        context = item['context']
        if args.rag > 0:
            template = template_rag
            retrieved = item["retrieved_context"][:args.rag]
            retrieved = sorted(retrieved, key=lambda x: x['c_idx'])
            context = '\n\n'.join([f"Retrieved chunk {idx+1}: {x['content']}" for idx, x in enumerate(retrieved)])
        elif args.no_context:
            template = template_no_context
        elif args.cot:
            template = template_0shot_cot
        else:
            template = template_0shot
        prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip())
        if args.cot:
            responses = query_llm([{"input": prompt, "outputs": ""}])
            # 取第一个响应
            output = responses[0]
        else:
            responses = query_llm([{"input": prompt, "outputs": ""}])
            # 取第一个响应
            output = responses[0]
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            responses = query_llm([{"input": prompt, "outputs": ""}])
            # 取第一个响应
            output = responses[0]
            if output == '':
                continue
        response = output.strip()
        item['response'] = response
        item['pred'] = extract_answer(response)
        item['judge'] = item['pred'] == item['answer']
        item['context'] = context[:1000]
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    if args.rag > 0:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + f"_rag_{str(args.rag)}" + args.appendix + ".jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + "_no_context" + args.appendix + ".jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + "_cot" + args.appendix + ".jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model_name.split("/")[-1] + args.appendix + ".jsonl")

    dataset = load_dataset('THUDM/LongBench-v2', split='train') # dataset = json.load(open('data.json', 'r', encoding='utf-8'))
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    # cache
    has_data = {}
    # if os.path.exists(out_file):
    #     with open(out_file, encoding='utf-8') as f:
    #         has_data = {json.loads(line)["_id"]: 0 for line in f}
    fout = open(out_file, 'w', encoding='utf-8')
    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)

    # data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
    # get_pred(data_subsets[0], args, fout)
    
    get_pred(data, args, fout)

    # processes = []
    # for rank in range(args.n_proc):
    #     p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    main()