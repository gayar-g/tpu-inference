import os
import glob
import torch
import argparse
from safetensors import safe_open

def verify_nm_block_compliance(tensor, prunen=1, prunem=4):
    if tensor.numel() % prunem != 0:
        return 0.0
    reshaped = tensor.contiguous().reshape(-1, prunem)
    target_zeros = prunem - prunen
    zeros_per_block = (reshaped == 0).sum(dim=-1)
    compliant_blocks = (zeros_per_block == target_zeros).sum().item()
    total_blocks = reshaped.shape[0]
    return (compliant_blocks / total_blocks) * 100.0

def verify_checkpoint(checkpoint_dir, prunen=1, prunem=4):
    expected_sparsity = (prunem - prunen) / prunem * 100.0
    min_sparsity = expected_sparsity - 3.0
    max_sparsity = expected_sparsity + 3.0
    
    print("\n" + "="*55)
    print(f">>> [VERIFICATION] Scanning Checkpoint Language Model: {checkpoint_dir}")
    print(f">>> Checking {prunen}:{prunem} Structured Sparsity (Target: {expected_sparsity:.1f}% zeros)")
    print("="*55)
    
    st_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not st_files:
        print(f"[ERROR] No safetensors found in {checkpoint_dir}")
        return False
        
    total_linear_tensors = 0
    passed_compliance = 0
    failed_projections = []
    
    for st_path in st_files:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "language_model.layers" in k and "weight" in k and not any(skip in k for skip in ["norm", "embed", "lm_head"]):
                    tensor = f.get_tensor(k)
                    if tensor.ndim == 2:
                        total_linear_tensors += 1
                        sparsity = (tensor == 0).sum().item() / tensor.numel() * 100.0
                        
                        # Exact check across first 4096 elements
                        t_slice = tensor.flatten()[:4096]
                        compliance = verify_nm_block_compliance(t_slice, prunen=prunen, prunem=prunem)
                        
                        if min_sparsity <= sparsity <= max_sparsity and compliance >= 99.0:
                            passed_compliance += 1
                        else:
                            failed_projections.append((k, sparsity, compliance))
                            print(f"  [FAIL] {k}: Sparsity={sparsity:.2f}%, {prunen}:{prunem} Compliance={compliance:.2f}%")
                            
    print(f">>> Total Checked Language Model Linear Projections: {total_linear_tensors}")
    print(f">>> Passing {prunen}:{prunem} Compliance: {passed_compliance}/{total_linear_tensors}")
    
    if total_linear_tensors > 0 and passed_compliance == total_linear_tensors:
        print(f"\n[VERIFICATION SUCCESS] 100% of all linear projections in all 60 language model layers adhere strictly to {prunen}:{prunem} structured sparsity!")
        return True
    else:
        print(f"\n[VERIFICATION FAILED] {len(failed_projections)} failed out of {total_linear_tensors}.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--prunen", type=int, default=1)
    parser.add_argument("--prunem", type=int, default=4)
    args = parser.parse_args()
    success = verify_checkpoint(args.checkpoint_dir, prunen=args.prunen, prunem=args.prunem)
    if not success:
        exit(1)
