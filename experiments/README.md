# TPU Inference & Structured Sparsity Benchmarking Suite

An automated, modular performance and accuracy evaluation framework for Large Language Models running on Google Cloud TPU v6e / v5e using vLLM and JAX.

---

## 1. System Architecture

The architecture adheres strictly to **SOLID design principles**, separating parameter grid generation, dynamic model resolution/splicing, polymorphic runner execution, and metric persistence.

```
                                  ┌────────────────────────────────┐
                                  │   run_sweep_orchestrator.py    │
                                  │   (Cartesian Grid Sequencer)   │
                                  └───────────────┬────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  ▼                                                               ▼
  ┌────────────────────────────────┐                             ┌────────────────────────────────┐
  │      ResolverRegistry          │                             │         RunnerRegistry         │
  │────────────────────────────────│                             │────────────────────────────────│
  │ • SparseGptResolver            │                             │ • TpuEvalRunner (lm_eval)      │
  │ • NaiveMagnitudeResolver       │                             │ • TpuProfileRunner (throughput)│
  │ • WeightScalerResolver         │                             └────────────────┬───────────────┘
  └───────────────┬────────────────┘                                              │
                  │ (resolves checkpoint path)                                    │ (executes benchmark)
                  ▼                                                               ▼
  ┌────────────────────────────────┐                             ┌────────────────────────────────┐
  │   Model Splicer & Pruners      │                             │     vLLM on Cloud TPU v6e      │
  │────────────────────────────────│                             │────────────────────────────────│
  │ • Cross-Shard ModelSplicer     │                             │ • 4-way Tensor Parallelism     │
  │ • TPU JAX SparseGPT Kernel     │                             │ • Batched RPA PagedAttention   │
  │ • Naive N:M Magnitude Pruner   │                             │ • FP8 KV Cache + bfloat16      │
  └────────────────────────────────┘                             └────────────────┬───────────────┘
                                                                                  │
                                                                                  ▼
                                                                 ┌────────────────────────────────┐
                                                                 │       CsvResultRecorder        │
                                                                 │   (results/<timestamp>/results.csv)
                                                                 └────────────────────────────────┘
```

---

## 2. Directory Structure

```
experiments/
├── configs/                  # YAML configurations for sweeps and benchmark runs
├── core/                     # Core interfaces, dataclasses, engine builders, and recorders
│   ├── interfaces.py         # IRunner, IModelResolver, IResultRecorder, ISparsityEngine
│   ├── models.py             # ExecutionContext, BenchmarkResult, ModelArchitectureSpec
│   ├── engine_builder.py     # vLLM kwargs builder and TPU environment setup
│   └── csv_recorder.py       # Dynamic schema-evolving CSV logger
├── resolvers/                # Dynamic checkpoint resolvers and layer transformers
│   ├── base_sparsity_resolver.py # Unified base class for sparsity resolvers
│   ├── sparsegpt_resolver.py     # SparseGPT Hessian-based resolver
│   ├── naive_magnitude_resolver.py # Data-free magnitude pruning resolver
│   ├── weight_scaler_resolver.py # Constant scalar multiplier (c * W) resolver
│   ├── model_splicer.py          # Cross-shard zero-recompute safetensors layer splicer
│   └── registry.py               # Central resolver factory
├── runners/                  # Benchmark execution engines
│   ├── eval_runner.py        # Accuracy evaluation runner (MMLU-Pro via lm_eval)
│   ├── profile_runner.py     # Latency & throughput profiling runner
│   ├── tpu_eval_vllm.py      # Standalone CLI evaluation script
│   ├── tpu_profile_vllm.py   # Standalone CLI profiling script
│   └── registry.py           # Central runner factory
├── sparsity/                 # Structured sparsity algorithms and TPU kernels
│   ├── tpu_sparsegpt.py      # JAX-accelerated SparseGPT kernel on TPU MXUs
│   ├── naive_magnitude_pruner.py # Vectorized N:M magnitude pruner (dim=-1)
│   └── gemma4.py             # Gemma 4 layer capture and Hessian orchestrator
├── run_sweep_orchestrator.py # Master sweep orchestrator
├── run_master_repeat_sweeps.py # Multi-experiment sequential runner
└── README.md
```

---

## 3. How to Sparsify Models Directly (Without `run_sweep_orchestrator`)

You can create 2:4 or 1:4 structured sparse checkpoints standalone without running a parameter sweep.

### A. TPU-Accelerated SparseGPT Pruning (Hessian-Compensated)
Runs second-order error-compensated pruning using calibration data (e.g. C4 dataset) directly on TPU Matrix Multiply Units:

```python
from sparsity import prune_gemma4

prune_gemma4(
    model_path="/mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475",
    dataset="c4",
    nsamples=256,              # Number of calibration sequences
    prunen=2,                  # N in N:M (2 for 2:4, 1 for 1:4)
    prunem=4,                  # M in N:M
    minlayer=0,                # Start layer index
    maxlayer=60,               # End layer index
    save_path="/mnt/pd/sparse_checkpoints/my_gemma4_2of4_256c4_full"
)
```

### B. Standalone Naive Magnitude Pruning (Data-Free)
Applies top-K magnitude pruning along the contracting dimension (dim=-1) across all linear projection weights without calibration:

```bash
python3 sparsity/naive_magnitude_pruner.py \
  --model /mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --save /mnt/pd/sparse_checkpoints/my_gemma4_2of4_magnitude_full \
  --prunen 2 \
  --prunem 4 \
  --minlayer 0 \
  --maxlayer 60
```

### C. Standalone Cross-Shard Layer Splicing
Splices specific layers from a 100% sparse reference model into a dense base model in seconds using `safe_open` zero-recompute streaming:

```python
from resolvers.model_splicer import ModelSplicer

splicer = ModelSplicer()
splicer.splice_checkpoint(
    dense_model_dir="/mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475",
    fully_sparse_model_dir="/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full",
    sparse_layer_indices={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},  # Splice first 10 layers
    output_dir="/mnt/pd/sparse_checkpoints/gemma4_spliced_10_layers"
)
```

### D. Standalone Constant Weight Scaling ($W \leftarrow c \cdot W$)
Scales only the 2D linear projection weights of a sparse model by a constant multiplier $c$ (e.g. $c = 1.10$):

```python
from resolvers.weight_scaler_resolver import WeightScalerResolver

resolver = WeightScalerResolver()
resolver._scale_checkpoint(
    base_checkpoint="/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full",
    output_dir="/mnt/pd/sparse_checkpoints/gemma4_2of4_scaled_c1p100",
    scale_factor=1.10
)
```

---

## 4. How to Run Sparse Models with vLLM on Cloud TPU

### A. Run Quality / Accuracy Evaluation (MMLU-Pro)
Evaluates any dense, sparse, or spliced checkpoint on MMLU-Pro via the `lm-evaluation-harness` across 4 TPU cores:

```bash
python3 runners/tpu_eval_vllm.py \
  --model /mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full \
  --tasks mmlu_pro \
  --limit 0.05 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --batch-size auto \
  --max-model-len 8192 

```

### B. Run Throughput & Latency Profiling
Measures generation throughput (tokens/second) and latency on TPU v6e:

```bash
python3 runners/tpu_profile_vllm.py \
  --model /mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full \
  --input-len 1024 \
  --output-len 128 \
  --batch-size 4 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --dtype bfloat16
```

---

## 5. How to Run Automated Sweeps

To execute a complete multi-coordinate parameter sweep (e.g. evaluating 0, 5, 10, 20, 30, 40, 50, 60 sparse layers):

```bash
python3 run_sweep_orchestrator.py --config configs/6_sparsity_front_to_back_2of4_mmlu_pro_eval_fast.yaml
```

**View Live Results:**
```bash
cat results/*/results.csv
```

**Resume an Interrupted Sweep:**
If a sweep was stopped or preempted, pass the previous experiment ID to resume without re-running finished coordinates:
```bash
python3 run_sweep_orchestrator.py --config configs/6_sparsity_front_to_back_2of4_mmlu_pro_eval_fast.yaml --experiment-id 20260819_183746
```
