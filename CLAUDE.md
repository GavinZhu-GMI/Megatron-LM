# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Megatron-LM is NVIDIA's production-grade library for distributed training of large language models at scale. It consists of two main components:

- **Megatron Core** (`megatron/core/`) - Composable, GPU-optimized library with modular building blocks for custom training frameworks
- **Megatron-LM** - Reference implementation demonstrating full training pipelines with pre-configured scripts

## Build and Installation

```bash
# Install with development dependencies (recommended)
pip install "setuptools<80.0.0,>=77.0.0" "packaging>=24.2"
pip install --no-build-isolation .[mlm,dev]

# Minimal install (torch + numpy only)
pip install megatron-core

# For LTS support (NGC PyTorch 24.01)
pip install --no-build-isolation megatron-core[mlm,lts]
```

**Note**: The `--no-build-isolation` flag is required due to the C++ extension (`megatron.core.datasets.helpers_cpp`) that requires pybind11.

## Testing

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run a specific test file
pytest tests/unit_tests/transformer/test_attention.py

# Run a single test
pytest tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_constructor -v

# Run with experimental features enabled
pytest --experimental tests/unit_tests/

# Run tests with coverage
pytest --cov=megatron tests/unit_tests/
```

Tests require GPU access and NCCL backend. Test data is auto-downloaded to `/opt/data` on first run.

## Linting and Formatting

Pre-commit hooks are configured for `megatron/core/` and `tests/unit_tests/`:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Individual tools (configured in pyproject.toml)
black --line-length 100 megatron/core/
isort --profile black megatron/core/
pylint megatron/core/
```

**Style Rules**:
- Line length: 100 characters
- Black with `--skip-string-normalization` and `--skip-magic-trailing-comma`
- isort with black profile
- Google-style docstrings

## Running Training

```bash
# Simple training example (2 GPUs, mock data)
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

# LLaMA-3 8B training (8 GPUs, FP8)
./examples/llama/train_llama3_8b_fp8.sh

# GPT pretraining
torchrun --nproc_per_node=8 pretrain_gpt.py [args]
```

## Architecture

### Core Module Structure (`megatron/core/`)

```
transformer/          # Composable building blocks (attention, MLP, transformer layers)
models/               # Model implementations (GPT, BERT, LLaMA, Mamba, MoE)
tensor_parallel/      # Tensor parallelism (column/row parallel patterns)
pipeline_parallel/    # Pipeline parallelism (schedules, virtual pipelines)
distributed/          # DDP, custom FSDP implementation
optimizer/            # Distributed optimizer, CPU offloading
datasets/             # Data loading, blended datasets, preprocessing
inference/            # Text generation, CUDA graphs, speculative decoding
dist_checkpointing/   # Sharded checkpoint save/load
parallel_state.py     # Process group management (TP, PP, DP, EP, CP)
```

### Parallelism Strategies

| Strategy | Flag | Description |
|----------|------|-------------|
| TP | `--tensor-model-parallel-size N` | Split layers across GPUs |
| PP | `--pipeline-model-parallel-size N` | Split depth across GPUs |
| DP | `--data-parallel-sharding-strategy` | Replicate with gradient aggregation (ZeRO-1/2/3) |
| CP | `--context-parallel-size N` | Split long sequences |
| EP | `--expert-model-parallel-size N` | MoE expert partitioning |

**Key constraint**: When combining EP with TP, sequence parallelism (`--sequence-parallel`) must be enabled.

### Entry Points

- `pretrain_gpt.py`, `pretrain_bert.py`, `pretrain_mamba.py` - Pre-training scripts
- `train_rl.py` - Reinforcement learning training
- `examples/run_simple_mcore_train_loop.py` - Minimal training example
- `tools/preprocess_data.py` - Data preprocessing for training

### Key Abstractions

- `TransformerConfig` - Central configuration dataclass for all transformer parameters
- `parallel_state` - Manages process groups; use `mpu` alias for backward compatibility
- `DistributedDataParallel` - Custom DDP with gradient accumulation handling
- `BlendedMegatronDatasetBuilder` - Multi-dataset mixing for training

## Data Preprocessing

```bash
python tools/preprocess_data.py \
    --input data.jsonl \
    --output-prefix processed_data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/tokenizer.model \
    --workers 8 \
    --append-eod
```

Input format is JSONL with `{"text": "..."}` per line.

## Performance Flags

| Feature | Flag | Purpose |
|---------|------|---------|
| FlashAttention | `--attention-backend` | Faster attention, lower memory |
| FP8 | `--fp8-hybrid` | Faster training on Hopper/Ada/Blackwell |
| Activation checkpointing | `--recompute-activations` | Reduce memory |
| Comm overlap | `--overlap-grad-reduce`, `--overlap-param-gather` | Hide communication |
| Distributed optimizer | `--use-distributed-optimizer` | Faster checkpointing |

## Docker (Recommended for Development)

```bash
docker run --runtime nvidia --gpus all -it --rm \
  -v /path/to/megatron:/workspace/megatron \
  -e PIP_CONSTRAINT= \
  nvcr.io/nvidia/pytorch:25.04-py3
```

Use the previous month's NGC container for compatibility with Megatron Core releases.
