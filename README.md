# entropix
Entropy Based Sampling and Parallel CoT Decoding

The goal is to replicate o1 style CoT with open source models

Current supported models:
  llama3.1+

Future supported models:
  DeepSeekV2+
  Mistral Large (123B)

# Getting Started
install poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

install rust to build tiktoken
```bash
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh
```

poetry install
```bash
poetry install
```

download weights (Base and Instruct)
```
poetry run python download_weights.py --model-id meta-llama/Llama-3.2-1B --out-dir weights/1B-Base
poetry run python download_weights.py --model-id meta-llama/Llama-3.2-1B-Instruct --out-dir weights/1B-Instruct
```

download tokenizer.model from huggingface (or wherever) into the entropix folder
e.g.
```bash
cd entropix
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --include "original/tokenizer.model" --local-dir ./llama3.1-tokenizer
```

run it
run it
```bash
 PYTHONPATH=. poetry run python entropix/main.py
```   
