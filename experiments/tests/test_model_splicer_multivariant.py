import os
import sys
import shutil
from typing import Set, Dict, Any
from safetensors import safe_open
from resolvers.model_splicer import ModelSplicer

DENSE_DIR = "/mnt/pd/huggingface_cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475"
SPARSE_DIR = "/mnt/pd/sparse_checkpoints/gemma4_31b_2of4_256c4_full"
TEST_VARIANTS = [5, 10, 30, 55, 60]
TOTAL_LAYERS = 60
TEMP_DIR_BASE = "/mnt/pd/sparse_checkpoints/test_splice_temp"

def verify_checkpoint_sparsity(ckpt_dir: str, target_sparse_count: int, expected_sparse_indices: Set[int]) -> Dict[str, Any]:
    layer_sparsity = {}
    
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(ckpt_dir, fname)
        with safe_open(fpath, framework="pt", device="cpu") as f:
            for k in f.keys():
                if "model.language_model.layers." in k and "weight" in k:
                    if not any(s in k for s in ["norm", "embed", "scalar"]):
                        parts = k.split("model.language_model.layers.")[1].split(".")
                        layer_idx = int(parts[0])
                        suffix = ".".join(parts[1:])
                        
                        t = f.get_tensor(k)
                        zero_ratio = (t == 0).sum().item() / t.numel()
                        if layer_idx not in layer_sparsity:
                            layer_sparsity[layer_idx] = {}
                        layer_sparsity[layer_idx][suffix] = zero_ratio

    assert len(layer_sparsity) == TOTAL_LAYERS, f"Expected {TOTAL_LAYERS} layers, found {len(layer_sparsity)}"

    detected_sparse_layers = set()
    detected_dense_layers = set()
    mismatches = []
    total_proj_count = 0

    for l_idx in range(TOTAL_LAYERS):
        projs = layer_sparsity.get(l_idx, {})
        total_proj_count += len(projs)
        
        assert len(projs) in [6, 7], f"Layer {l_idx} unexpected projection count: {len(projs)}"
        
        is_sparse = all(abs(r - 0.50) < 1e-4 for r in projs.values())
        is_dense = all(r == 0.0 for r in projs.values())

        if is_sparse:
            detected_sparse_layers.add(l_idx)
        elif is_dense:
            detected_dense_layers.add(l_idx)
        else:
            mismatches.append(f"Layer {l_idx} has partial/corrupted sparsity: {projs}")

        if l_idx in expected_sparse_indices and not is_sparse:
            mismatches.append(f"Layer {l_idx} expected SPARSE (50%), but got is_sparse={is_sparse}, projs={projs}")
        elif l_idx not in expected_sparse_indices and not is_dense:
            mismatches.append(f"Layer {l_idx} expected DENSE (0%), but got is_dense={is_dense}, projs={projs}")

    return {
        "target_count": target_sparse_count,
        "sparse_layers_found": len(detected_sparse_layers),
        "dense_layers_found": len(detected_dense_layers),
        "total_projections_verified": total_proj_count,
        "mismatches": mismatches,
        "success": len(detected_sparse_layers) == target_sparse_count and len(mismatches) == 0
    }

def run_all_tests():
    print("=" * 85)
    print(">>> STARTING MULTI-VARIANT SPLICING TEST FOR GEMMA 4 31B")
    print(f">>> Variants to test: {TEST_VARIANTS} layers")
    print(f">>> Dense Source:    {DENSE_DIR}")
    print(f">>> Sparse Source:   {SPARSE_DIR}")
    print("=" * 85)

    summary = []

    for num_sparse in TEST_VARIANTS:
        out_dir = f"{TEMP_DIR_BASE}_{num_sparse}_layers"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        sparse_indices = set(range(num_sparse))
        print("\n--------------------------------------------------------------------------------")
        print(f"[TEST VARIANT: {num_sparse} Sparse Layers (Layers 0..{num_sparse - 1})]")
        print("--------------------------------------------------------------------------------")
        
        # 1. Splice
        ModelSplicer.splice_checkpoint(
            dense_model_dir=DENSE_DIR,
            fully_sparse_model_dir=SPARSE_DIR,
            sparse_layer_indices=sparse_indices,
            output_dir=out_dir
        )

        # 2. Verify
        result = verify_checkpoint_sparsity(
            ckpt_dir=out_dir,
            target_sparse_count=num_sparse,
            expected_sparse_indices=sparse_indices
        )

        tgt = result["target_count"]
        sp_f = result["sparse_layers_found"]
        dn_f = result["dense_layers_found"]
        tot_l = sp_f + dn_f
        tot_p = result["total_projections_verified"]
        status = "PASSED" if result["success"] else "FAILED"

        print(f">>> Verification Result for {num_sparse} layers:")
        print(f"    - Target Sparse Layers:       {tgt}")
        print(f"    - Sparse Layers Verified:     {sp_f}")
        print(f"    - Dense Layers Verified:      {dn_f}")
        print(f"    - Total Layers Verified:      {tot_l} / {TOTAL_LAYERS}")
        print(f"    - Total Projections Verified: {tot_p} / 410")
        print(f"    - Status:                     {status}")

        if not result["success"]:
            for m in result["mismatches"]:
                print(f"      [ERROR] {m}")

        summary.append(result)

        # 3. Clean up
        shutil.rmtree(out_dir)
        print(f">>> Cleaned temporary checkpoint: {out_dir}")

    print("\n" + "=" * 85)
    print(">>> FINAL MULTI-VARIANT SPLICING TEST SUMMARY")
    print("=" * 85)
    hdr_target = "Target Sparse Layers"
    hdr_sparse = "Sparse Found"
    hdr_dense = "Dense Found"
    hdr_projs = "Projs Verified"
    hdr_status = "Status"
    print(f"{hdr_target:<22} | {hdr_sparse:<14} | {hdr_dense:<14} | {hdr_projs:<16} | {hdr_status:<10}")
    print("-" * 85)
    all_passed = True
    for r in summary:
        tgt = r["target_count"]
        sp_f = r["sparse_layers_found"]
        dn_f = r["dense_layers_found"]
        tot_p = r["total_projections_verified"]
        status_str = "PASSED" if r["success"] else "FAILED"
        if not r["success"]:
            all_passed = False
        print(f"{tgt:<22} | {sp_f:<14} | {dn_f:<14} | {tot_p:<16} | {status_str:<10}")
    print("=" * 85)

    if all_passed:
        print(">>> ALL 5 MULTI-VARIANT SPLICING TESTS PASSED PERFECTLY (100% VERIFIED)!\n")
    else:
        print(">>> SOME TESTS FAILED!\n")
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
