"""
run_sweep_orchestrator.py
-------------------------
Modular TPU Sweep Orchestrator for throughput profiling and benchmark accuracy evaluation.

Adheres to SOLID architecture principles:
- SRP: Generates parameter cartesian grids and delegates execution to runners, resolvers, and recorders.
- OCP: Extensible via RunnerRegistry and ResolverRegistry without modifying the orchestration loop.
- LSP: Polymorphic execution through IRunner and IModelResolver interfaces.
- ISP: Clean interface separation via ExecutionContext and BenchmarkResult dataclasses.
- DIP: Decoupled orchestration depending strictly on abstractions.
"""

import os
import sys
import yaml
import json
import csv
import itertools
import argparse
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple, Optional

# Ensure experiments root directory is in the Python search path
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from core.models import ExecutionContext, BenchmarkResult
from core.csv_recorder import CsvResultRecorder
from runners.registry import RunnerRegistry
from resolvers.registry import ResolverRegistry

DEFAULT_RESULT_DIR = os.path.join(EXPERIMENTS_DIR, "results")

class SweepOrchestrator:
    """
    Main orchestrator for managing multi-dimensional parameter sweeps on Cloud TPUs.
    Coordinates parameter grid expansion, model resolution, runner execution, and metric persistence.
    """

    def __init__(self, config_path: str, result_dir: str = DEFAULT_RESULT_DIR, experiment_id: Optional[str] = None):
        """Initializes the sweep orchestrator and prepares output directory and CSV recorder."""
        self.config_path = config_path
        self.result_dir = result_dir
        self.experiment_id = experiment_id
        self.config = self._load_config(config_path)

        exp_id = self.experiment_id if self.experiment_id else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(self.result_dir, exp_id)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.csv_file = os.path.join(self.exp_dir, "results.csv")
        # Initialize CSV result recorder
        self.recorder = CsvResultRecorder(self.csv_file)

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Loads and parses the YAML experiment configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _resolve_model(self, current_config: Dict[str, Any]) -> Tuple[str, Set[str]]:
        """Resolves target model path using ResolverRegistry or falls back to static model path."""
        spec = self.config.get("model_resolver")
        
        # For normal inference parameter sweeps without sparsity or modifying model, we can just return the model path from the config
        if not spec:
            return current_config.get("model", self.config.get("model", "")), set()

        # Otherwise if we need to resolve a model dynamically (e.g., sparsity, pruning, etc.), we use the resolver registry
        resolver = ResolverRegistry.get(spec.get("module", "sparsegpt"))
        return resolver.resolve(current_config, spec.get("config", {}))

    def _get_completed_coordinates(self) -> List[Dict[str, Any]]:
        """Reads existing rows from the results CSV to support seamless sweep resumption."""
        completed: List[Dict[str, Any]] = []
        if os.path.exists(self.csv_file):
            with open(self.csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.append(row)
        return completed

    def run(self) -> str:
        """Executes the full sweep matrix across all Cartesian product coordinates."""
        mode = self.config.get("mode", "profile").lower()
        sweep = self.config.get("sweep_matrix", {})
        eval_cfg = self.config.get("evaluation", {})

        # Normalize sweep matrix values into lists
        for k, v in sweep.items():
            if not isinstance(v, list):
                sweep[k] = [v]

        sweep_keys = list(sweep.keys())
        sweep_values = list(sweep.values())
        cross_product = list(itertools.product(*sweep_values))

        # Check for previously completed coordinates
        completed_configs = self._get_completed_coordinates()

        print(f"=======================================================")
        print(f">>> [SWEEP ORCHESTRATOR] Initializing Sweep: {self.config_path}")
        print(f">>> Total Grid Points: {len(cross_product)} | Mode: {mode.upper()}")
        print(f">>> Experiment Directory: {self.exp_dir}")
        print(f"=======================================================")

        for combo in cross_product:
            current_config = dict(zip(sweep_keys, combo))

            # Resumption check: skip coordinate if already recorded in CSV
            skip = False
            for completed_row in completed_configs:
                if all(str(completed_row.get(k)) == str(v) for k, v in current_config.items()):
                    skip = True
                    break

            if skip:
                print(f">>> [SKIPPED] Coordinate {current_config} already completed in {self.csv_file}.")
                continue

            # 1. Resolve target model checkpoint dynamically
            target_model, consumed_keys = self._resolve_model(current_config)

            # 2. Build normalized ExecutionContext
            combined_metadata = {**current_config}
            batch_size = current_config.get("batch_size", eval_cfg.get("batch_size", self.config.get("batch_size", "auto")))
            tasks = current_config.get("tasks", eval_cfg.get("tasks", "mmlu_pro"))
            limit = current_config.get("limit", eval_cfg.get("limit", None))

            context = ExecutionContext(
                model_path=target_model,
                tensor_parallel_size=self.config.get("tensor_parallel_size", 4),
                dtype=self.config.get("dtype", "bfloat16"),
                max_model_len=self.config.get("max_model_len", None),
                engine_args=self.config.get("engine_args", {}),
                sweep_metadata=combined_metadata,
                output_dir=self.exp_dir,
                csv_file=self.csv_file,
                batch_size=batch_size,
                input_len=current_config.get("input_len", current_config.get("inputs", 128)),
                output_len=current_config.get("output_len", current_config.get("output", 64)),
                trace=bool(self.config.get("trace_configs", [])),
                tasks=tasks,
                limit=limit
            )

            # 3. Execute benchmark via polymorphic runner (profile, eval, or both)
            runner_names = ["profile", "eval"] if mode == "both" else [mode]
            for r_name in runner_names:
                runner = RunnerRegistry.get(r_name)
                result: BenchmarkResult = runner.execute(context)

                # 4. Persist trial metrics to CSV
                cmd_repro = f"python3 run_sweep_orchestrator.py --config {self.config_path}"
                self.recorder.record(
                    metadata=combined_metadata,
                    metrics=result.metrics,
                    reproduction_cmd=cmd_repro
                )

        print(f">>> [SWEEP COMPLETE] Results written to {self.csv_file}")
        return self.csv_file

def main():
    """CLI entry point for running parameter sweeps."""
    parser = argparse.ArgumentParser(description="Modular TPU Sweep Orchestrator")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help="Base directory for results")
    parser.add_argument("--experiment-id", default=None, help="Pass to resume sweep from specific ID")
    args = parser.parse_args()

    orchestrator = SweepOrchestrator(
        config_path=args.config,
        result_dir=args.result_dir,
        experiment_id=args.experiment_id
    )
    orchestrator.run()

if __name__ == "__main__":
    main()
