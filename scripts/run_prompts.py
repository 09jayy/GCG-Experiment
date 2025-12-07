import time
import csv
import argparse

from functions.prompt_model import run_parallel_prompts

def defined_default_params():
    return {
        "input": "../data/harmful_prompts.txt",  
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "range": None
    }

def configure_run_arguments():
    default_params = defined_default_params()
    parser = argparse.ArgumentParser(
        description="Run prompts through LLM model without GCG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i","--input", default=default_params["input"])
    parser.add_argument("-m","--model",help="hugging face model id", default=default_params["model_id"])
    parser.add_argument("-r","--range",nargs=2,help="set range for sub list of list of input prompts",default=default_params["range"])

    args = parser.parse_args()
    input_prompts = []
    with open(args.input) as f:
        for line in f:
            if line[0] != "#": input_prompts.append(line.strip("\n"))
    args.input = input_prompts if args.range is None else input_prompts[int(args.range[0]):int(args.range[1])]
    return args

if __name__ == "__main__":
    args = configure_run_arguments()

    results = run_parallel_prompts(args.input, args.model, num_workers=8)

    time_string = time.strftime("%Y-%m-%d_%H:%M:%S")
    with open(f'results/prompt_results_{time_string}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt', 'llm_response'])
        
        for i, result in enumerate(results):
            print(f"=====\n{i}: {result}\n=====")
            
            # Write row immediately (flushes to disk)
            writer.writerow([result[0], result[1]])
            f.flush()  # Force write to disk after each row

    print(f"Results saved to results/prompt_results_{time_string}.csv")

    