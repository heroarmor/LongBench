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

sys.path.append(os.path.abspath(os.path.join (os.path.dirname(__file__), "..")))
from model.llama_anns import LlamaForCausalLM
from utils.kv_database import VectorDBCache
from transformers.cache_utils import OffloadedCache

# set random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

model_map = json.loads(open('config/model2path.json', encoding='utf-8').read())
maxlen_map = json.loads(open('config/model2maxlen.json', encoding='utf-8').read())

URL = "http://127.0.0.1:8000/v1"
API_KEY = "token-abc123"
template_rag = open('prompts/0shot_rag.txt', encoding='utf-8').read()
template_no_context = open('prompts/0shot_no_context.txt', encoding='utf-8').read()
template_0shot = open('prompts/0shot.txt', encoding='utf-8').read()
template_0shot_cot = open('prompts/0shot_cot.txt', encoding='utf-8').read()
template_0shot_cot_ans = open('prompts/0shot_cot_ans.txt', encoding='utf-8').read()

def query_llm(prompt, model_name, model, tokenizer, client=None, temperature=0.5, max_new_tokens=128, stop=None):
    # truncate
    max_len = maxlen_map[model_name]
    if model_name in model_map:
        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)
    else:
        input_ids = tokenizer.encode(prompt, disallowed_special=())
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids)
    tries = 0

    if model_name in model_map:
        model_name = model_map[model_name]
    
    while tries < 5:
        tries += 1
        try:
            # completion = client.chat.completions.create(
            #     model=model,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=temperature,
            #     max_tokens=max_new_tokens,
            # )
            # return completion.choices[0].message.content
            messages = [{"role": "user", "content": prompt}]
            tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            print(tokenized_chat)
            encoded = tokenizer(tokenized_chat, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
            inputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"]
            }
            print(inputs["input_ids"].shape)
            # kv_cache = OffloadedCache ()
            kv_cache = VectorDBCache(use_anns=True, index="faiss_ivf_flat")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    past_key_values=kv_cache,
                    use_cache=True,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            content = tokenizer.decode(outputs[0], skip_special_tokens=False)
            print(content)
            return content
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return ''

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
    model_name = args.model
    #if "gpt" in model_name or "o1" in model_name:
        #tokenizer = tiktoken.encoding_for_model("gpt-4o-2024-08-06")
    #else:
    tokenizer = AutoTokenizer.from_pretrained(model_map[model_name], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # model = AutoModelForCausalLM.from_pretrained(model_map[model_name]).to (torch.bfloat16).to("cuda:0")
    model = LlamaForCausalLM.from_pretrained(model_map[model_name]).to (torch.bfloat16).to("cuda:0")

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
            output = query_llm(prompt, model_name, model, tokenizer, client, temperature=0.1, max_new_tokens=1024)
        else:
            output = query_llm(prompt, model_name, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
        if output == '':
            continue
        if args.cot: # extract answer
            response = output.strip()
            item['response_cot'] = response
            prompt = template_0shot_cot_ans.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip()).replace('$C_A$', item['choice_A'].strip()).replace('$C_B$', item['choice_B'].strip()).replace('$C_C$', item['choice_C'].strip()).replace('$C_D$', item['choice_D'].strip()).replace('$COT$', response)
            output = query_llm(prompt, model_name, model, tokenizer, client, temperature=0.1, max_new_tokens=128)
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
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + f"_rag_{str(args.rag)}" + args.appendix + ".jsonl")
    elif args.no_context:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_no_context" + args.appendix + ".jsonl")
    elif args.cot:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + "_cot" + args.appendix + ".jsonl")
    else:
        out_file = os.path.join(args.save_dir, args.model.split("/")[-1] + args.appendix + ".jsonl")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    parser.add_argument("--appendix", "-a", type=str, default="") # set to "" if not using appendix, otherwise set to the appendix name
    args = parser.parse_args()
    main()