# GCG-Experiment

Test Large Language Model Jailbreak Attack Success Rate using GCG (Greedy Coordiante Gradient). Runs GCG and LLM Prompts using parallel processing via CUDA. 

# Requirements

- HuggingFace
- PyTorch
- Transformers 

# Instructions

```bash
git clone https://github.com/09jayy/GCG-Experiment
cd GCG-Experiment
pip install -e .
```

## Run GCG Experiment Script

```bash
python3 scripts/run_gcg.py 

# see run optional arguments and default values
python3 scripts/run_gcg.py --help

```

## Run Prompts to LLM Model

```bash
python3 scripts/run_prompts.py

# optional run arguments and default values
python3 scripts/run_prompts.py --help
``` 

# References

**GCG Algorithm:**
Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. and Fredrikson, M. (n.d.). Universal and Transferable Adversarial Attacks on Aligned Language Models. [online] Available at: https://arxiv.org/pdf/2307.15043.