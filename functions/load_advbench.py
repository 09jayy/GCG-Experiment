import csv

def load_advbench(dir="./data/advbench.csv"):
    prompts = []
    targets = []
    with open(dir, mode = "r") as f:
        csv_file = csv.DictReader(f)
        for row in csv_file:
            prompts.append(row["goal"])
            targets.append(row["target"])
    return prompts,targets

if __name__ == "__main__":
    prompts, targets = load_advbench()
    for pair in zip(prompts,targets):
        print(pair)