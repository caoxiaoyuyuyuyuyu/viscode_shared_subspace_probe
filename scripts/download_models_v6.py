#!/usr/bin/env python
"""Stage 0 v6 model download — D008 W6 + D011 NW2 compliant.

Strict rules:
- HF_HOME only + HF_ENDPOINT mirror + HF_HUB_DISABLE_XET
- No local_dir, no cache_dir (let HF use default HF_HOME/hub/ layout)
- max_workers=8, etag_timeout=300s
- Size-ascending download order (sbert -> CLIP -> VisCoder2)
- D011 NW2: smoke test immediately after each download, fail-fast
- Fallback endpoint: if hf-mirror.com fails 3 times, switch to huggingface.co
"""
import os, sys, time, traceback

# D008 W6: HF_HOME + mirror + XET disable
os.environ['HF_HOME'] = '/root/autodl-tmp/.hf_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
# Wipe conflicting envs
for k in ['TRANSFORMERS_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ.pop(k, None)

from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForCausalLM
import torch

LOG = '/tmp/model_download_v6.log'
FALLBACK_AFTER_FAILS = 3

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def disk_free_gb():
    import shutil
    return shutil.disk_usage('/root/autodl-tmp').free / 1e9

# Size-ascending; D011 NW2 order
MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", 0.022, AutoModel),
    ("openai/clip-vit-large-patch14",           0.428, AutoModel),
    ("TIGER-Lab/VisCoder2-7B",                  7.6,   AutoModelForCausalLM),
]

def try_snapshot(repo_id):
    return snapshot_download(
        repo_id=repo_id,
        max_workers=8,
        etag_timeout=300,
        allow_patterns=[
            '*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*',
            'special_tokens_map.json', '*.py', 'vocab*', 'merges*',
            'preprocessor_config.json', '*.bin.index.json',
        ],
    )

def download_with_fallback(repo_id):
    fails = 0
    endpoints = ['https://hf-mirror.com', 'https://huggingface.co']
    ep_idx = 0
    while ep_idx < len(endpoints):
        os.environ['HF_ENDPOINT'] = endpoints[ep_idx]
        log(f"  endpoint={endpoints[ep_idx]} attempt")
        try:
            path = try_snapshot(repo_id)
            log(f"  downloaded to {path}")
            return path
        except Exception as e:
            fails += 1
            log(f"  FAIL ({type(e).__name__}): {str(e)[:300]}")
            if fails >= FALLBACK_AFTER_FAILS and ep_idx < len(endpoints) - 1:
                ep_idx += 1
                fails = 0
                log(f"  switching endpoint -> {endpoints[ep_idx]}")
                continue
            if ep_idx >= len(endpoints) - 1 and fails >= FALLBACK_AFTER_FAILS:
                log(f"  all endpoints exhausted after {fails} fails")
                raise
            time.sleep(5 + 5 * fails)
    raise RuntimeError(f"{repo_id}: all endpoints exhausted")

def smoke_test(repo_id, expected_params_B, model_cls):
    log(f"  smoke test via repo_id {repo_id}")
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        m = model_cls.from_pretrained(
            repo_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        p_B = sum(x.numel() for x in m.parameters()) / 1e9
        dev_pct = abs(p_B - expected_params_B) / expected_params_B * 100
        log(f"  params={p_B:.4f}B expected~{expected_params_B}B dev={dev_pct:.1f}%")
        if dev_pct > 5:
            log(f"  SMOKE_FAIL dev >5%")
            return False, p_B
        log(f"  SMOKE_PASS model_type={m.config.model_type}")
        del m
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True, p_B
    finally:
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)

def main():
    log("=== Stage 0 v6 download start ===")
    log(f"HF_HOME={os.environ['HF_HOME']}")
    log(f"HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
    log(f"disk_free_initial={disk_free_gb():.1f}GB")

    passed = []
    for repo_id, expected_B, cls in MODELS:
        log(f"--- {repo_id} ---")
        free = disk_free_gb()
        log(f"disk_free_before={free:.1f}GB")
        try:
            download_with_fallback(repo_id)
        except Exception as e:
            log(f"DOWNLOAD_FAIL {repo_id}: {e}")
            log(f"passed_so_far={passed}")
            sys.exit(1)
        ok, params = smoke_test(repo_id, expected_B, cls)
        if not ok:
            log(f"FAIL {repo_id} smoke test")
            log(f"passed_so_far={passed}")
            sys.exit(1)
        passed.append((repo_id, params))
        log(f"disk_free_after={disk_free_gb():.1f}GB")
        os.system('sync')

    log(f"=== ALL OK === passed={passed}")

if __name__ == '__main__':
    main()
