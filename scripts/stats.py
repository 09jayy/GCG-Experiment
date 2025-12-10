import pandas as pd
import argparse

def configure_run_arguments():
    parser = argparse.ArgumentParser(
        description="Run GCG (using nanogcg library) in parallel using CUDA device GPU",
        epilog="Example: python3 main.py", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-i","--input",help="input path file for prompts") 
    args = parser.parse_args()

    return args

if __name__ == "__main__": 
    args = configure_run_arguments() 

    df = pd.read_csv(args.input)

    print(df.get("gcg_safe").value_counts())
    print(df.get("raw_safe").value_counts())

"""
output e.g.

gcg_safe
unsafe    407
safe      113
Name: count, dtype: int64
raw_safe
safe      376
unsafe    144
Name: count, dtype: int64

"""