"""
Standalone quality and accuracy evaluation script for vLLM on TPUs.

Evaluates HuggingFace and local checkpoints on standard benchmarks (such as MMLU-Pro,
GSM8K, etc.) using the lm_eval harness powered by vLLM on TPU devices.
Results and benchmark scores are appended to a central CSV alongside the reproduction command.
"""

import argparse
import os
import sys
import json
import csv
import subprocess
from datetime import datetime


def main(args):
    print(f"\n=======================================================")
    print(f">>> [EVALUATION] Initializing Model Evaluation")
    print(f">>> Model Checkpoint: {args.model}")
    print(f">>> Tasks:            {args.tasks}")
    print(f">>> Sample Limit:     {args.limit}")
    print(f"=======================================================\n")

    # Output directory for lm_eval raw JSON log dumps
    eval_logs_dir = os.path.join(args.eval_result_dir, "eval_logs")
    os.makedirs(eval_logs_dir, exist_ok=True)

    # Construct vLLM engine arguments for lm_eval
    engine_kwargs = {
        "pretrained": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "disable_chunked_mm_input": True,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.85
    }
    if args.max_model_len:
        engine_kwargs["max_model_len"] = args.max_model_len

    if args.engine_args:
        extra_args = json.loads(args.engine_args)
        engine_kwargs.update(extra_args)

    model_args_str = json.dumps(engine_kwargs)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args_str,
        "--tasks", args.tasks,
        "--apply_chat_template",
        "--batch_size", str(args.batch_size),
        "--output_path", eval_logs_dir,
        "--seed", "42"
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    # Setup pure XLA orchestration environment
    env = os.environ.copy()
    env["VLLM_TARGET_DEVICE"] = "tpu"
    env["VLLM_PLATFORM"] = "tpu"
    env["USE_BATCHED_RPA_KERNEL"] = "1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    start_time = datetime.now()
    subprocess.run(cmd, env=env, check=True)
    end_time = datetime.now()
    duration_s = (end_time - start_time).total_seconds()

    # Locate and parse the latest JSON evaluation result
    latest_json = None
    latest_mtime = 0
    for root, _, files in os.walk(eval_logs_dir):
        for f in files:
            if f.endswith(".json") and "results" in f:
                fpath = os.path.join(root, f)
                mtime = os.path.getmtime(fpath)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_json = fpath

    parsed_scores = {}
    if latest_json:
        with open(latest_json, "r") as jf:
            eval_data = json.load(jf)
            if "results" in eval_data:
                for task_name, task_metrics in eval_data["results"].items():
                    for metric_name, val in task_metrics.items():
                        if isinstance(val, (int, float)):
                            parsed_scores[f"{task_name}_{metric_name}"] = f"{val:.4f}"

    print("\n========== Benchmark Evaluation Results ==========")
    meta = {}
    if args.sweep_metadata:
        meta = json.loads(args.sweep_metadata)
    else:
        meta = {"model": args.model, "tasks": args.tasks, "limit": args.limit}

    for k, v in meta.items():
        print(f"{k.capitalize():<25} {v}")

    for k, v in parsed_scores.items():
        print(f"{k:<25} {v}")

    print(f"{'Eval_Duration:':<25} {duration_s:.2f} s")
    print("==================================================\n")

    # Record to target CSV
    csv_file = args.csv_file if args.csv_file else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "summary_metrics.csv")
    csv_dir = os.path.dirname(csv_file)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    meta["Eval_Duration_s"] = f"{duration_s:.2f}"
    meta.update(parsed_scores)
    meta["Reproduction_Command"] = f"python3 {' '.join(sys.argv)}"

    file_exists = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    if file_exists:
        with open(csv_file, "r") as f_read:
            try:
                fieldnames = next(csv.reader(f_read))
            except StopIteration:
                fieldnames = list(meta.keys())
    else:
        fieldnames = list(meta.keys())

    # Add any new metric fields dynamically to fieldnames
    for key in meta.keys():
        if key not in fieldnames:
            fieldnames.append(key)

    # Re-read existing rows and rewrite CSV with updated header if fieldnames expanded
    if file_exists:
        with open(csv_file, "r") as f_read:
            existing_rows = list(csv.DictReader(f_read))
        with open(csv_file, "w", newline="") as f_write:
            writer = csv.DictWriter(f_write, fieldnames=fieldnames, extrasaction='ignore', restval='')
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(meta)
    else:
        with open(csv_file, "w", newline="") as f_write:
            writer = csv.DictWriter(f_write, fieldnames=fieldnames, extrasaction='ignore', restval='')
            writer.writeheader()
            writer.writerow(meta)

    print(f"Wrote evaluation metrics to {csv_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--tasks", type=str, default="mmlu_pro_philosophy", help="Comma-separated lm_eval tasks (e.g. mmlu_pro,gsm8k)")
    parser.add_argument("--batch-size", default="auto", help="Batch size for lm_eval")
    parser.add_argument("--limit", type=float, default=None, help="Sample limit per task (useful for quick smoke tests)")
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size across TPU devices")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum context length")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model weight/activation dtype")
    parser.add_argument("--eval-result-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    parser.add_argument("--csv-file", type=str, default=None, help="Explicit path to target CSV file")
    parser.add_argument("--engine-args", type=str, default="{}", help="JSON string of extra kwargs for vLLM")
    parser.add_argument("--sweep-metadata", type=str, default="{}", help="JSON dict outlining precise configuration for logging")
    args = parser.parse_args()
    main(args)
