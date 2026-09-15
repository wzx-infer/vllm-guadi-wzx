#!/usr/bin/env python3
"""
Unified vLLM Gaudi benchmark runner.
Usage:
    python run_bench.py --config configs/base_smoke.json
    python run_bench.py --config configs/scene_sweep.json --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def log(msg, log_file=None):
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_cmd(cmd, log_file=None, check=True):
    log(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}", log_file)
    proc = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True
    )
    if proc.stdout:
        print(proc.stdout, flush=True)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(proc.stdout + "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(proc.stderr + "\n")
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {cmd}")
    return proc


def wait_server_ready(api_url, api_key, timeout=600, interval=10, log_file=None):
    """Poll /v1/models until server is up."""
    import urllib.request
    models_url = api_url.rsplit("/v1/", 1)[0] + "/v1/models"
    log(f"Waiting for server: {models_url}", log_file)
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(models_url)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    log("Server is ready.", log_file)
                    return True
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"Server not ready after {timeout}s")


def build_docker_cmd(cfg, run_params, container_name):
    """Build docker run command from config + current run params."""
    model = cfg["model"]
    server = cfg["server"]
    env = dict(cfg.get("env", {}))
    # override / add per-run env
    env.update(run_params.get("env", {}))

    cmd = ["docker", "run", "-d", "--name", container_name]
    # privileged + devices
    cmd += ["--privileged", "-v", "/dev/accel:/dev/habanalabs"]
    # env vars
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    # port
    cmd += ["-p", f"{server['port']}:{server['port']}"]
    # volumes
    for vol in cfg.get("volumes", []):
        cmd += ["-v", vol]
    # runtime flags
    cmd += ["--ipc=host", "--cap-add=sys_nice", "--security-opt", "label=disable"]
    # image + entrypoint
    cmd += ["--entrypoint", "python3", cfg["docker_image"]]
    # vllm args
    cmd += ["-m", "vllm.entrypoints.openai.api_server"]
    cmd += ["--model", model["path"]]
    cmd += ["--served-model-name", model["name"]]
    cmd += ["--dtype", model.get("dtype", "bfloat16")]
    if model.get("quantization"):
        cmd += ["--quantization", model["quantization"]]
    cmd += ["--tensor-parallel-size", str(model.get("tensor_parallel_size", 4))]
    cmd += ["--block-size", str(model.get("block_size", 128))]
    if model.get("enable_expert_parallel", True):
        cmd += ["--enable-expert-parallel"]
    # per-run vllm args override
    vllm_args = dict(model.get("vllm_args", {}))
    vllm_args.update(run_params.get("vllm_args", {}))
    for k, v in vllm_args.items():
        if isinstance(v, bool):
            if v:
                cmd += [f"--{k}"]
        else:
            cmd += [f"--{k}", str(v)]
    cmd += ["--port", str(server["port"])]
    return cmd


def build_evalscope_cmd(cfg, case):
    """Build evalscope perf command from config + test case."""
    model = cfg["model"]
    api = cfg["api"]
    cmd = [
        "evalscope", "perf",
        "--parallel", str(case["parallel"]),
        "--number", str(case["number"]),
        "--warmup-num", str(case["warmup"]),
        "--model", model["name"],
        "--url", api["url"],
        "--api", "openai",
        "--dataset", "random",
        "--min-prompt-length", str(case["in_len"]),
        "--max-prompt-length", str(case["in_len"]),
        "--max-tokens", str(case["out_len"]),
        "--tokenizer-path", model["tokenizer_path"],
        "--api-key", api["key"],
        "--temperature", "0.0",
        "--total-timeout", str(cfg.get("total_timeout", 3600)),
    ]
    if case.get("type") != "prefill_only":
        cmd += ["--min-tokens", str(case["out_len"])]
    return cmd


def expand_cases(raw_cases, cfg):
    """Expand case templates with auto number/warmup logic."""
    cases = []
    for c in raw_cases:
        parallel = c["parallel"]
        in_len = c["in_len"]
        number = c.get("number")
        if number is None:
            if parallel <= 4:
                number = 16
            elif parallel <= 16:
                number = 32
            else:
                number = 64
        warmup = c.get("warmup")
        if warmup is None:
            if in_len >= 32768:
                warmup = 8
            elif in_len >= 8192:
                warmup = 6
            else:
                warmup = 4
        cases.append({
            "type": c.get("type", "mixed"),
            "parallel": parallel,
            "in_len": in_len,
            "out_len": c["out_len"],
            "number": number,
            "warmup": warmup,
        })
    return cases


def run_single_container_mode(cfg, log_file):
    """Mode 1 & 2: start container once, run all cases."""
    container_name = cfg["server"]["container_name"]
    run_params = {"env": cfg.get("run_env", {}), "vllm_args": cfg.get("run_vllm_args", {})}

    # cleanup old
    run_cmd(["docker", "rm", "-f", container_name], check=False)
    # start
    docker_cmd = build_docker_cmd(cfg, run_params, container_name)
    run_cmd(docker_cmd, log_file)
    # wait
    wait_server_ready(cfg["api"]["url"], cfg["api"]["key"],
                       timeout=cfg.get("server_ready_timeout", 600), log_file=log_file)
    # run cases
    cases = expand_cases(cfg["cases"], cfg)
    for i, case in enumerate(cases, 1):
        log(f"\n===== Case {i}/{len(cases)}: {case} =====", log_file)
        evals_cmd = build_evalscope_cmd(cfg, case)
        run_cmd(evals_cmd, log_file, check=cfg.get("fail_fast", True))
        time.sleep(cfg.get("case_cooldown", 5))
    # cleanup
    if cfg.get("cleanup_container", True):
        run_cmd(["docker", "rm", "-f", container_name], check=False)


def run_param_search_mode(cfg, log_file):
    """Mode 3: one container per param combination."""
    container_base = cfg["server"]["container_name"]
    param_matrix = cfg["param_matrix"]
    fixed_case = cfg["fixed_case"]

    results = []
    for idx, combo in enumerate(param_matrix, 1):
        label = combo.get("label", f"combo_{idx}")
        container_name = f"{container_base}_{label}"
        log(f"\n{'='*60}\nParam Search {idx}/{len(param_matrix)}: {label}\n{combo}\n{'='*60}", log_file)

        run_params = {"env": combo.get("env", {}), "vllm_args": combo.get("vllm_args", {})}
        run_cmd(["docker", "rm", "-f", container_name], check=False)
        docker_cmd = build_docker_cmd(cfg, run_params, container_name)
        try:
            run_cmd(docker_cmd, log_file)
            wait_server_ready(cfg["api"]["url"], cfg["api"]["key"],
                               timeout=cfg.get("server_ready_timeout", 600), log_file=log_file)
            case = expand_cases([fixed_case], cfg)[0]
            evals_cmd = build_evalscope_cmd(cfg, case)
            proc = run_cmd(evals_cmd, log_file, check=False)
            results.append({"label": label, "combo": combo, "rc": proc.returncode})
        except Exception as e:
            log(f"FAILED: {e}", log_file)
            results.append({"label": label, "combo": combo, "error": str(e)})
        finally:
            run_cmd(["docker", "rm", "-f", container_name], check=False)
            time.sleep(cfg.get("combo_cooldown", 10))

    # summary
    summary_path = Path(log_file).parent / "param_search_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Param search summary saved to {summary_path}", log_file)


def main():
    parser = argparse.ArgumentParser(description="Unified vLLM Gaudi benchmark runner")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--log-dir", default=None, help="Override log directory")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    log_dir = Path(args.log_dir or cfg.get("log_dir", ROOT / "perf_logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{cfg_path.stem}_{stamp}.log"

    mode = cfg.get("mode", "single_container")
    log(f"Config: {cfg_path}")
    log(f"Mode: {mode}")
    log(f"Log: {log_file}")

    if args.dry_run:
        log("DRY RUN - no commands will be executed")
        # preview first docker cmd and first evalscope cmd
        if mode == "param_search":
            combo = cfg["param_matrix"][0]
            run_params = {"env": combo.get("env", {}), "vllm_args": combo.get("vllm_args", {})}
        else:
            run_params = {"env": cfg.get("run_env", {}), "vllm_args": cfg.get("run_vllm_args", {})}
        docker_cmd = build_docker_cmd(cfg, run_params, cfg["server"]["container_name"])
        log("Docker cmd preview: " + " ".join(docker_cmd))
        case_src = cfg.get("fixed_case") or cfg["cases"][0]
        case = expand_cases([case_src], cfg)[0]
        evals_cmd = build_evalscope_cmd(cfg, case)
        log("Evalscope cmd preview: " + " ".join(evals_cmd))
        return

    if mode == "param_search":
        run_param_search_mode(cfg, log_file)
    else:
        run_single_container_mode(cfg, log_file)

    log("All done.")


if __name__ == "__main__":
    main()
