"""
engine_builder.py
-----------------
Helper class responsible for constructing vLLM engine kwargs and configuring
TPU-specific environment variables for high-performance JAX/Pallas inference.
"""

import os
from typing import Dict, Any
from core.models import ExecutionContext

class VllmEngineBuilder:
    """
    Constructs vLLM runtime engine configuration dictionaries and sets up
    TPU platform environment variables.
    """

    @staticmethod
    def setup_tpu_environment() -> None:
        """Sets required environment variables for vLLM TPU execution."""
        os.environ["VLLM_TARGET_DEVICE"] = "tpu"
        os.environ["VLLM_PLATFORM"] = "tpu"
        os.environ["USE_BATCHED_RPA_KERNEL"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    @staticmethod
    def build_engine_kwargs(context: ExecutionContext) -> Dict[str, Any]:
        """
        Assembles normalized vLLM engine arguments from the execution context.
        Sets memory utilization, parallel dimensions, and precision defaults.
        """
        engine_kwargs = {
            "tensor_parallel_size": context.tensor_parallel_size,
            "dtype": context.dtype,
            "disable_chunked_mm_input": True,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": 0.85
        }
        # Apply max model context length if specified
        if context.max_model_len:
            engine_kwargs["max_model_len"] = context.max_model_len

        # Merge custom engine args from YAML sweep configuration
        if context.engine_args:
            engine_kwargs.update(context.engine_args)

        return engine_kwargs
