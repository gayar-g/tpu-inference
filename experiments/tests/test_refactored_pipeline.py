import os
import sys
import unittest
import tempfile
import csv

# Add experiments directory to path
EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENTS_DIR)

from core.models import ExecutionContext, BenchmarkResult, ModelArchitectureSpec
from core.engine_builder import VllmEngineBuilder
from core.csv_recorder import CsvResultRecorder
from runners.registry import RunnerRegistry
from resolvers.model_splicer import ModelSplicer
from resolvers.sparsegpt_resolver import SparseGptResolver
from resolvers.registry import ResolverRegistry


class TestRefactoredPipeline(unittest.TestCase):
    def test_engine_builder(self):
        ctx = ExecutionContext(
            model_path="dummy/path",
            tensor_parallel_size=4,
            dtype="bfloat16",
            max_model_len=8192,
            engine_args={"kv_cache_dtype": "fp8"}
        )
        kwargs = VllmEngineBuilder.build_engine_kwargs(ctx)
        self.assertEqual(kwargs["tensor_parallel_size"], 4)
        self.assertEqual(kwargs["dtype"], "bfloat16")
        self.assertEqual(kwargs["max_model_len"], 8192)
        self.assertEqual(kwargs["kv_cache_dtype"], "fp8")
        self.assertTrue(kwargs["disable_chunked_mm_input"])

    def test_csv_recorder_schema_expansion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "results.csv")
            recorder = CsvResultRecorder(csv_path)

            # Record initial row
            recorder.record(
                metadata={"num_sparse_layers": 0},
                metrics={"mmlu_pro_exact_match": 0.8586},
                reproduction_cmd="python3 test.py"
            )

            # Record second row with extra new metric keys
            recorder.record(
                metadata={"num_sparse_layers": 5},
                metrics={"mmlu_pro_exact_match": 0.7977, "gsm8k_acc": 0.9120},
                reproduction_cmd="python3 test.py"
            )

            # Verify file exists and has expanded columns
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(len(rows), 2)
                self.assertIn("gsm8k_acc", reader.fieldnames)
                self.assertEqual(rows[0]["num_sparse_layers"], "0")
                self.assertEqual(rows[0]["gsm8k_acc"], "")
                self.assertEqual(rows[1]["num_sparse_layers"], "5")
                self.assertEqual(rows[1]["gsm8k_acc"], "0.912")

    def test_layer_index_parsing(self):
        indices = ModelSplicer.parse_layer_indices_str("0-4,10,55-59", total_layers=60)
        expected = {0, 1, 2, 3, 4, 10, 55, 56, 57, 58, 59}
        self.assertEqual(indices, expected)

    def test_sparsegpt_resolver_layer_ranges(self):
        resolver = SparseGptResolver()
        # Last to first (5 layers out of 60 -> 55 to 60)
        start, end = resolver.compute_layer_range(5, "last_to_first", 60)
        self.assertEqual((start, end), (55, 60))

        # First to last (10 layers out of 60 -> 0 to 10)
        start, end = resolver.compute_layer_range(10, "first_to_last", 60)
        self.assertEqual((start, end), (0, 10))

        # Middle out (10 layers out of 60 -> 25 to 35)
        start, end = resolver.compute_layer_range(10, "middle_out", 60)
        self.assertEqual((start, end), (25, 35))

    def test_runner_registry(self):
        profile_runner = RunnerRegistry.get("profile")
        self.assertIsNotNone(profile_runner)

        eval_runner = RunnerRegistry.get("eval")
        self.assertIsNotNone(eval_runner)

        with self.assertRaises(KeyError):
            RunnerRegistry.get("non_existent_runner")

    def test_resolver_registry(self):
        resolver = ResolverRegistry.get("sparsegpt")
        self.assertIsInstance(resolver, SparseGptResolver)


if __name__ == "__main__":
    unittest.main()
