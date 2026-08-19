"""
eval_runner.py
--------------
Quality and accuracy evaluation runner using the lm-evaluation-harness on Cloud TPUs.
Executes standard benchmarks (e.g. MMLU-Pro) via vLLM and parses output metrics.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from typing import Dict, Any

from core.models import ExecutionContext, BenchmarkResult
from core.engine_builder import VllmEngineBuilder
from runners.base_runner import BaseVllmRunner

class TpuEvalRunner(BaseVllmRunner):
    """
    Executes accuracy and quality evaluation via the lm-evaluation-harness on TPU v6e.
    Runs lm_eval as an isolated subprocess to manage memory and device cleanup.
    """

    def execute(self, context: ExecutionContext) -> BenchmarkResult:
        """
        Launches lm_eval for the given ExecutionContext, parses metric results,
        and returns a normalized BenchmarkResult.
        """
        print("\n=======================================================")
        print(f">>> [EVALUATION] Model: {context.model_path}")
        print(f">>> Tasks: {context.tasks} | Limit: {context.limit} | Batch Size: {context.batch_size}")
        print("=======================================================")

        # Create output directory for lm_eval JSON results
        eval_logs_dir = os.path.join(context.output_dir, "eval_logs")
        os.makedirs(eval_logs_dir, exist_ok=True)

        # Build vLLM engine kwargs and serialize to JSON string
        engine_kwargs = VllmEngineBuilder.build_engine_kwargs(context)
        engine_kwargs["pretrained"] = context.model_path
        model_args_str = json.dumps(engine_kwargs)

        # Assemble CLI invocation for lm_eval
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "vllm",
            "--model_args", model_args_str,
            "--tasks", context.tasks,
            "--apply_chat_template",
            "--batch_size", str(context.batch_size),
            "--output_path", eval_logs_dir,
            "--seed", str(context.seed)
        ]
        if context.limit is not None:
            cmd.extend(["--limit", str(context.limit)])

        # Set up TPU environment variables for the child process
        env = os.environ.copy()
        env["VLLM_TARGET_DEVICE"] = "tpu"
        env["VLLM_PLATFORM"] = "tpu"
        env["USE_BATCHED_RPA_KERNEL"] = "1"
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Execute evaluation and track wall-clock duration
        start_time = datetime.now()
        subprocess.run(cmd, env=env, check=True)
        end_time = datetime.now()
        duration_s = (end_time - start_time).total_seconds()

        # Locate the most recently generated results JSON file
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

        # Parse task accuracy scores and standard errors from JSON
        parsed_scores: Dict[str, Any] = {}
        if latest_json:
            with open(latest_json, "r") as jf:
                eval_data = json.load(jf)
                if "results" in eval_data:
                    for task_name, task_metrics in eval_data["results"].items():
                        for metric_name, val in task_metrics.items():
                            if isinstance(val, (int, float)):
                                parsed_scores[f"{task_name}_{metric_name}"] = f"{val:.4f}"

        metrics = {
            "Eval_Duration_s": f"{duration_s:.2f}",
            **parsed_scores
        }
        print(f">>> [EVALUATION] Complete: Duration {duration_s:.2f}s")
        return BenchmarkResult(duration_s=duration_s, metrics=metrics, log_path=latest_json)
