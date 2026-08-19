"""
profile_runner.py
------------------
Throughput, latency, and performance profiling runner for vLLM on Cloud TPUs.
Executes warmup/timed generation cycles and captures optional JAX XProf traces.
"""

import os
import time
from typing import Dict, Any, List
from vllm import LLM, SamplingParams
import jax

from core.models import ExecutionContext, BenchmarkResult
from core.engine_builder import VllmEngineBuilder
from runners.base_runner import BaseVllmRunner

class TpuProfileRunner(BaseVllmRunner):
    """
    Profiles throughput (tokens/sec) and generation latency on TPU v6e.
    Supports JAX XProf hardware performance counter tracing.
    """

    def execute(self, context: ExecutionContext) -> BenchmarkResult:
        """
        Initializes the LLM engine, warms up TPU kernels, runs timed inference,
        and computes throughput metrics.
        """
        engine_kwargs = VllmEngineBuilder.build_engine_kwargs(context)
        engine_kwargs["model"] = context.model_path

        print(f">>> [PROFILING] Initializing vLLM engine: {engine_kwargs}")
        llm = LLM(**engine_kwargs)

        # Generate synthetic input prompts matching batch_size and input_len
        prompts = ["Hello " * context.input_len] * context.batch_size
        sampling_params = SamplingParams(max_tokens=context.output_len, temperature=0.0)

        # 1. Warmup step to compile JAX computation graphs and TPU kernels
        print(">>> [PROFILING] Warming up JAX/TPU compilation graphs...")
        _ = llm.generate(prompts=prompts[:1], sampling_params=sampling_params)

        # 2. Optional JAX Performance Counter Profiling Trace
        if context.trace:
            options = jax.profiler.ProfileOptions()
            options.host_tracer_level = 2
            options.device_tracer_level = 1
            if context.jax_advanced_configuration:
                options.advanced_configuration = context.jax_advanced_configuration

            trace_dir = os.path.join(context.output_dir, "traces")
            os.makedirs(trace_dir, exist_ok=True)
            print(f">>> [PROFILING] Starting JAX profiler trace in {trace_dir}...")
            jax.profiler.start_trace(trace_dir, profiler_options=options)

        # 3. Timed generation benchmark
        start_time = time.time()
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
        end_time = time.time()

        # Stop profiling trace if active
        if context.trace:
            jax.profiler.stop_trace()
            print(">>> [PROFILING] Stopped JAX profiler trace.")

        # Compute throughput and latency metrics
        duration_s = end_time - start_time
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        throughput_tok_s = total_tokens / duration_s if duration_s > 0 else 0.0

        metrics = {
            "Duration_s": f"{duration_s:.4f}",
            "Throughput_tok_s": f"{throughput_tok_s:.2f}"
        }

        print(f">>> [PROFILING] Complete: {throughput_tok_s:.2f} tok/s in {duration_s:.4f}s")
        return BenchmarkResult(duration_s=duration_s, metrics=metrics)
