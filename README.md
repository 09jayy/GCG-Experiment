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
```text
@misc{zou2023universal,
      title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, 
      author={Andy Zou and Zifan Wang and J. Zico Kolter and Matt Fredrikson},
      year={2023},
      eprint={2307.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```