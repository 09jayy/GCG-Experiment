import torch
import time
import argparse

from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import csv
import multiprocessing as mp
from functions.GCG_parallel import run_parallel_gcg
from functions.prompt_model import run_parallel_prompts

def defined_default_params():
    return {
        "input_json": "data/harmful_prompts.txt",  
        "steps": 300,
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "range": None,
        "run_gcg": True
    }

def configure_run_arguments():
    default_params = defined_default_params()
    parser = argparse.ArgumentParser(
        description="Run GCG (using nanogcg library) in parallel using CUDA device GPU",
        epilog="Example: python3 main.py", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i","--input",help="input path .json file for prompts", default=default_params["input_json"])
    parser.add_argument("-s","--steps",help="number of steps for each prompt that gcg is run for", type=int,default=default_params["steps"])
    parser.add_argument("-m","--model", help="hugging face model id",default=default_params["model_id"]) 
    parser.add_argument("-r","--range",nargs=2,help="set range for sub list of list of input prompts",default=default_params["range"])

    args = parser.parse_args()
    print(args.range)

    input_prompts = []
    with open(args.input) as f:
        for line in f:
            if line[0] != "#": input_prompts.append(line.strip("\n"))
    args.input = input_prompts if args.range is None else input_prompts[int(args.range[0]):int(args.range[1])]
    return args

def run_gcg(model_id, harmful_prompts):
    config = GCGConfig(
        num_steps=args.steps,
        search_width=64,
        topk=64,
        seed=42,
        verbosity="WARNING",
        optim_str_init = "x x x x x x x x x x"
    )

    results = []
    results = run_parallel_gcg(harmful_prompts, model_id, config, num_workers=8)
    return results

if __name__ == "__main__":
    args = configure_run_arguments()

    # PARAMETERS
    model_id = args.model
    
    # LLM Params    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    torch.use_deterministic_algorithms(False)
    
    results = run_gcg(model_id, args.input)
    llm_responses = run_parallel_prompts([x[0]+x[1] for x in results], model_id,num_workers=8)

    time_string = time.strftime("%Y-%m-%d_%H:%M:%S")
    with open(f'results/gcg_results_{time_string}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'adv_suffix', 'llm_response', 'best_loss'])
        
        for result,llm_response in zip(results,llm_responses):
            prompt, adv_suffix, loss = result
            _, response,_ = llm_response 
            
            print(f"=====\n{prompt}\n=====")
            print(f"response")
            
            # Write row immediately (flushes to disk)
            writer.writerow([prompt, adv_suffix, response, loss])
            f.flush()  # Force write to disk after each row

    print(f"Results saved to results/gcg_results_{time_string}.csv")