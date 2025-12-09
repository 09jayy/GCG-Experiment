import argparse
import csv 
import os
import time
from scripts.run_gcg import run_gcg
from functions.prompt_model import run_parallel_prompts
from functions.load_advbench import load_advbench

def defined_default_params():
    return {
        "input_json": "data/harmful_prompts.txt",  
        "steps": 300,
        "model_id": "Qwen/Qwen2-0.5B-Instruct",
        "range": None,
        "run_gcg": True,
        "num_workers": 6
    }

def configure_run_arguments():
    default_params = defined_default_params()
    parser = argparse.ArgumentParser(
        description="Run GCG (using nanogcg library) in parallel using CUDA device GPU",
        epilog="Example: python3 main.py", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-s","--steps",help="number of steps for each prompt that gcg is run for", type=int,default=default_params["steps"])
    parser.add_argument("-m","--model", help="hugging face model id",default=default_params["model_id"]) 
    parser.add_argument("-w","--num-workers",help="number of workers for parallel processing, depends on GPU memory capacity",type=int, default=default_params["num_workers"])
    parser.add_argument("-r","--range",nargs=2,help="set range for sub list of list of input prompts",type=int,default=default_params["range"])

    args = parser.parse_args() 

    return args


if __name__ == "__main__":
    args = configure_run_arguments()

    # load advbench dataset from GCG paper
    advbench_prompts, advbench_targets = load_advbench()
    
    # reduce dataset is range given in runtime arguments
    if args.range is not None:
        advbench_prompts = advbench_prompts[args.range[0]:args.range[1]]
        advbench_targets = advbench_targets[args.range[0]:args.range[1]]

    # run gcg
    adv_prompts = run_gcg(args.model, advbench_prompts, args.steps, args.num_workers, advbench_targets)
    # run LLM with GCG
    adv_results = run_parallel_prompts([x[0]+x[1] for x in adv_prompts], args.model,args.num_workers)

    # run LLM without GCG
    raw_results = run_parallel_prompts(advbench_prompts, args.model, args.num_workers)

    # add gcg response , and prompt response to .csv file
    time_string = time.strftime("%Y-%m-%d_%H:%M:%S")
    os.makedirs("results/", exist_ok=True)
    with open(f"results/compare_results_{time_string}.csv", 'w', newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "adv_suffix", "gcg_response","raw_response","best_loss"])

        for adv_result, raw_result, adv_prompt in zip(adv_results, raw_results, adv_prompts):
            # extract prompt and adv_suffix
            prompt, adv_suffix, loss = adv_prompt
            # extract adverarial response
            _,adv_response,_ = adv_result
            # extract raw response (without adversarial suffix)
            _,raw_response,_ = raw_result

            writer.writerow([prompt,adv_suffix,adv_response,raw_response,loss])
            f.flush() # force write
    print(f"Results saved to: results/compare_results_{time_string}.csv")
