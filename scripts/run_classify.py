import csv
import argparse
import os
import time
from functions.prompt_guard import run_parallel_safety

def configure_run_arguments():
    parser = argparse.ArgumentParser(
        description="Classify LLM responses",
        epilog="Example: python3 run_classify.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i","--input",help="input path file")

    return parser.parse_args()

if __name__ == "__main__": 
    args = configure_run_arguments() 

    time_string = time.strftime("%Y-%m-%d_%H:%M:%S")
    # input shape: prompt, adv_suffix, gcg_response, raw_response, best_loss
    # get columns for gcg and raw responses
    with open(args.input, 'r') as r:
        reader = csv.reader(r)
        gcg_responses = []
        raw_responses = []
        for prompt, adv_suffix, gcg_response, raw_response, best_loss in reader:
            gcg_responses.append(gcg_response)
            raw_responses.append(raw_response)

    # get safety classification of GCG prompts
    gcg_safe = run_parallel_safety(gcg_responses)
    raw_safe = run_parallel_safety(raw_responses)

    with open(args.input, 'r') as r:
        with open(f"results/classify_results_{time_string}.csv", 'w',newline='',encoding="utf-8") as w:
            reader = csv.reader(r)
            next(reader)
            writer = csv.writer(w)
            writer.writerow(["prompt", "adv_suffix", "gcg_response", "raw_response", "best_loss", "gcg_safe", "raw_safe"])
            for compare_result,(_,gcg_safe_class,_),(_,raw_safe_class,_) in zip(reader,gcg_safe,raw_safe):
                writer.writerow([*compare_result,gcg_safe_class, raw_safe_class])
                w.flush()
    print(f"Results saved to: results/classify_results_{time_string}.csv")
                