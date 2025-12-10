import torch
import time
import argparse
import os

from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
from functions.GCG_parallel import run_parallel_gcg
from functions.load_advbench import load_advbench
from functions.prompt_model import run_parallel_prompts
from tabulate import tabulate

def defined_default_params():
    return {
        "input": "data/harmful_prompts.txt",  
        "steps": 300,
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "range": None,
        "run_gcg": True,
        "num_workers": 7
    }

def configure_run_arguments():
    default_params = defined_default_params()
    parser = argparse.ArgumentParser(
        description="Run GCG (using nanogcg library) in parallel using CUDA device GPU",
        epilog="Example: python3 main.py", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i","--input",help="input path file for prompts", default=default_params["input"])
    parser.add_argument("-s","--steps",help="number of steps for each prompt that gcg is run for", type=int,default=default_params["steps"])
    parser.add_argument("-m","--model", help="hugging face model id",default=default_params["model_id"]) 
    parser.add_argument("-w","--num-workers",help="number of workers for parallel processing, depends on GPU memory capacity",type=int, default=default_params["num_workers"])
    parser.add_argument("-r","--range",nargs=2,help="set range for sub list of list of input prompts",type=int,default=default_params["range"])

    args = parser.parse_args() 

    if args.input == "advbench":
        args.input,_ = load_advbench()
        return args

    # Ignore lines in .txt files where line starts with '#' character
    input_prompts = []
    if args.input.split('.')[-1] == ".txt":
        with open(args.input) as f:
            for line in f:
                if line[0] != "#": input_prompts.append(line.strip("\n"))
        args.input = input_prompts if args.range is None else input_prompts[int(args.range[0]):int(args.range[1])]

    return args

def run_gcg(model_id, harmful_prompts, steps, num_workers, targets=None) -> list[tuple[str, str ,float]]:
    config = GCGConfig(
        num_steps=steps,
        search_width=64,
        topk=64,
        seed=42,
        verbosity="WARNING",
        optim_str_init = "x x x x x x x x x x"
    )

    results = []
    results = run_parallel_gcg(harmful_prompts, model_id, config, num_workers=num_workers, targets=targets)
    return results

if __name__ == "__main__":
    args = configure_run_arguments()

    # PARAMETERS
    model_id = args.model
    
    # run gcg for all prompts
    start = time.perf_counter()
    results = run_gcg(model_id, args.input, args.steps, args.num_workers)
    gcg_timedelta = time.perf_counter() - start
    
    # run llm with prompts + adversial suffix
    start = time.perf_counter()
    llm_responses = run_parallel_prompts([x[0]+x[1] for x in results], model_id,num_workers=args.num_workers)
    llm_responses_timedelta = time.perf_counter() - start

    time_string = time.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs("results/", exist_ok=True)
    with open(f'results/gcg_results_{time_string}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'adv_suffix', 'llm_response', 'best_loss'])
        
        for result,llm_response in zip(results,llm_responses):
            prompt, adv_suffix, loss = result
            _, response,_ = llm_response 
            
            # Write row immediately (flushes to disk)
            writer.writerow([prompt, adv_suffix, response, loss])
            f.flush()  # Force write to disk after each row

    print(f"Results saved to: results/gcg_results_{time_string}.csv")
    print(tabulate([["GCG",gcg_timedelta],["LLM Response",llm_responses_timedelta]], headers=["Task", "Time Taken"]))

