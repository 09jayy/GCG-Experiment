import time
import csv
import argparse
import os
from tabulate import tabulate

from functions.prompt_model import run_parallel_prompts

def defined_default_params():
    return {
        "input": "../data/harmful_prompts.txt",  
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "range": None,
        "num_workers": 7 
    }

def configure_run_arguments():
    default_params = defined_default_params()
    parser = argparse.ArgumentParser(
        description="Run prompts through LLM model without GCG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i","--input", default=default_params["input"])
    parser.add_argument("-m","--model",help="hugging face model id", default=default_params["model_id"])
    parser.add_argument("-r","--range",nargs=2,help="set range for sub list of list of input prompts",type=int,default=default_params["range"])
    parser.add_argument("-w","--num-workers",help="number of workers for parallel processing",type=int,default=default_params["num_workers"])

    args = parser.parse_args()

    # Ignore lines in .txt files where line starts with '#' character
    input_prompts = []
    with open(args.input) as f:
        for line in f:
            if line[0] != "#": input_prompts.append(line.strip("\n"))
    args.input = input_prompts if args.range is None else input_prompts[args.range[0]:args.range[1]]

    return args

if __name__ == "__main__":
    args = configure_run_arguments()

    start = time.perf_counter()
    results = run_parallel_prompts(args.input, args.model, num_workers=args.num_workers)
    prompts_timedelta = time.perf_counter() - start

    time_string = time.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs("results/", exist_ok=True)
    with open(f'results/prompt_results_{time_string}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'llm_response'])
        
        for result in results:        
            # Write row immediately (flushes to disk)
            writer.writerow([result[0], result[1]])
            f.flush()  # Force write to disk after each row

    print(f"Results saved to: results/prompt_results_{time_string}.csv")
    print(tabulate([["LLM Responses", prompts_timedelta]],headers=["Task", "Time Taken"]))