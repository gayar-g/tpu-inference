import yaml
import subprocess
import argparse
import os
import json
import itertools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, "tpu_profile_vllm.py")
DEFAULT_RESULT_DIR = os.path.join(BASE_DIR, "results")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help="Base directory for results")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model = config.get("model", "google/gemma-4-12b-it")
    tp_size = config.get("tensor_parallel_size", 4)
    dtype = config.get("dtype", "bfloat16")
    max_model_len = config.get("max_model_len", None)
    env_vars = config.get("env_vars", {})
    engine_args = config.get("engine_args", {})
    
    sweep = config.get("sweep_matrix", {})
    trace_configs = config.get("trace_configs", [])

    # Backward compatibility mapping for older yaml formats
    if "batches" in sweep: sweep["batch_size"] = sweep.pop("batches")
    elif "batch_size" not in sweep: sweep["batch_size"] = config.get("batches", [1])
    
    if "inputs" in sweep: sweep["input_len"] = sweep.pop("inputs")
    elif "input_lens" in sweep: sweep["input_len"] = sweep.pop("input_lens")
    elif "input_len" not in sweep: sweep["input_len"] = config.get("inputs", [128])
    
    if "output_lens" in sweep: sweep["output_len"] = sweep.pop("output_lens")
    elif "output_len" not in sweep: sweep["output_len"] = config.get("output_len", [64])

    # Enforce all values as iterables natively 
    for k, v in sweep.items():
        if not isinstance(v, list):
            sweep[k] = [v]

    sweep_keys = list(sweep.keys())
    sweep_values = list(sweep.values())
    cross_product = list(itertools.product(*sweep_values))

    # Pre-Flight Validation for trace_configs
    for tc in trace_configs:
        # Normalize tc keys
        canonical_tc = {}
        for k, v in tc.items():
            if k == "jax_advanced_configuration": continue
            if k == "batch": k = "batch_size"
            if k in ["input", "inputs"]: k = "input_len"
            canonical_tc[k] = v
            
        for k, v in canonical_tc.items():
            if k not in sweep_keys or v not in sweep[k]:
                print(f"❌ ERROR: Trace config {tc} is invalid. {k}={v} is not in the generated sweep matrix boundaries.")
                return

    # Clean model directory name and CSV file path
    model_dir_name = model.replace("/", "--")
    yaml_stem = os.path.splitext(os.path.basename(args.config))[0]
    model_result_dir = os.path.join(args.result_dir, model_dir_name)
    csv_file = os.path.join(model_result_dir, f"{yaml_stem}.csv")
    os.makedirs(model_result_dir, exist_ok=True)
    
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write("Model,Batch_Size,Input_Len,Output_Len,Duration_s,Throughput_tok_s,Reproduction_Command\n")
    
    for combo in cross_product:
        current_config = dict(zip(sweep_keys, combo))
        b = current_config.get("batch_size", 1)
        i = current_config.get("input_len", 128)
        o = current_config.get("output_len", 64)
        
        skip = False
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                for line in f:
                    if line.startswith(f"{model},{b},{i},{o},"):
                        skip = True
                        break
                        
        if skip:
            print(f">>> [SKIPPED] Batch {b} | Input {i} | Output {o} already exists in {csv_file}.")
            continue
            
        print(f">>> Running Batch {b} | Input {i} | Output {o}")
        cmd = [
            "/mnt/pd/shen/vllm_env/bin/python3", SCRIPT_PATH,
            "--model", model,
            "--csv-file", csv_file,
            "--tensor-parallel-size", str(tp_size),
            "--dtype", dtype,
            "--engine-args", json.dumps(engine_args)
        ]
        
        # Build CLI args generically
        for k, val in current_config.items():
            cmd.extend([f"--{k.replace('_', '-')}", str(val)])
            
        if max_model_len:
            cmd.extend(["--max-model-len", str(max_model_len)])
            
        # Target isolated JAX Traces
        for tc in trace_configs:
            # Normalize user trace matching to coordinate dictionary explicitly
            canonical_tc = {}
            for k, v in tc.items():
                if k == "jax_advanced_configuration": continue
                if k == "batch": k = "batch_size"
                if k in ["input", "inputs"]: k = "input_len"
                canonical_tc[k] = v
                
            # If everything in canonical_tc matches current_config natively
            if all(current_config.get(k) == v for k, v in canonical_tc.items()):
                trace_name = "_".join(f"{k}{v}" for k, v in canonical_tc.items())
                trace_dir = os.path.join(model_result_dir, f"trace_{trace_name}")
                cmd.extend(["--trace", "--profile-result-dir", trace_dir])
                if "jax_advanced_configuration" in tc:
                    cmd.extend(["--jax-advanced-configuration", json.dumps(tc["jax_advanced_configuration"])])
                print(f"    >>> Profiling trace enabled! Dumps routing to {trace_dir}")
                break

        # Setup pure XLA orchestration environment
        env = os.environ.copy()
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        for k, v in env_vars.items():
            env[str(k)] = str(v)
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"!!! CRASH DETECTED at Batch {b} | Input {i} | Output {o}. This usually indicates a limits Wall.")
            print(f"!!! Cleaning TPU locks and safely moving to the next matrix dimension...")
            subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"])
                
    print(f"✅ Sweep configuration {args.config} complete! Results written to {csv_file}")

if __name__ == "__main__":
    main()
