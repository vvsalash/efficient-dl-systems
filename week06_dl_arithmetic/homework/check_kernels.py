import argparse
import os
import re
import subprocess
import sys
import tempfile


def count_kernels(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return len(set(re.findall(r"def (triton_[A-Za-z0-9_]+)", text)))


def count_graph_breaks(path: str) -> int:
    if not os.path.exists(path):
        return -1
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()
    return text.count("graph break") + text.count("graph_break")


def run_step(target: str, dtype: str) -> None:
    import torch
    import torch.nn as nn

    device = "cuda"
    torch_dtype = getattr(torch, dtype)
    model = nn.Sequential(*[nn.Linear(256, 256, bias=False) for _ in range(4)])
    model = model.to(device=device, dtype=torch_dtype)

    if target == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        name = "AdamW"
    else:
        from efficient_optimizer.ademamix import AdEMAMix

        opt = AdEMAMix(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999, 0.9999),
            alpha=2.0,
            alpha_warmup=None,
            beta3_warmup=None,
            weight_decay=0.1,
        )
        name = "AdEMAMix"

    for p in model.parameters():
        p.grad = torch.randn_like(p)

    torch.compile(opt.step, mode="reduce-overhead", fullgraph=False, dynamic=False)()

    log_path = os.environ["TORCH_LOGS_OUT"]
    print(f"{name} graph breaks: {count_graph_breaks(log_path)}")
    print(f"{name} Triton kernels: {count_kernels(log_path)}")


def run_subprocess(target: str, dtype: str, log_path: str):
    env = os.environ.copy()
    env["TORCH_LOGS_OUT"] = log_path
    env.setdefault("TORCHINDUCTOR_DISABLE_CUDAGRAPHS", "1")

    cmd = [sys.executable, __file__, "--target", target, "--dtype", dtype]
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--target", default="both", choices=["both", "adamw", "ademamix"])
    args = parser.parse_args()

    if "TORCH_LOGS" not in os.environ or "+output_code" not in os.environ["TORCH_LOGS"]:
        raise RuntimeError("Set TORCH_LOGS=+output_code,+graph_breaks")
    if "TORCH_LOGS_OUT" not in os.environ:
        raise RuntimeError("Set TORCH_LOGS_OUT=/path/to/log")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.target != "both":
        run_step(args.target, args.dtype)
        return

    base_log = os.environ["TORCH_LOGS_OUT"]
    log_dir = tempfile.mkdtemp(prefix="opt_kernels_")
    adamw_log = os.path.join(log_dir, "adamw.log")
    ademamix_log = os.path.join(log_dir, "ademamix.log")

    adamw_proc = run_subprocess("adamw", args.dtype, adamw_log)
    ademamix_proc = run_subprocess("ademamix", args.dtype, ademamix_log)

    os.environ["TORCH_LOGS_OUT"] = base_log

    print("=== AdamW run ===")
    print(f"return code: {adamw_proc.returncode}")
    print(adamw_proc.stdout.strip())
    if adamw_proc.returncode != 0:
        print(adamw_proc.stderr.strip())

    print("=== AdEMAMix run ===")
    print(f"return code: {ademamix_proc.returncode}")
    print(ademamix_proc.stdout.strip())
    if ademamix_proc.returncode != 0:
        print(ademamix_proc.stderr.strip())

    adamw_kernels = count_kernels(adamw_log)
    ademamix_kernels = count_kernels(ademamix_log)

    if adamw_kernels > 0 and ademamix_kernels > 0:
        if ademamix_kernels <= adamw_kernels:
            print("PASS: AdEMAMix kernel count is not greater than AdamW")
        else:
            print("FAIL: AdEMAMix kernel count is greater than AdamW")
    else:
        print("WARNING: Missing kernel evidence from one or both runs")

    print(f"AdamW log: {adamw_log}")
    print(f"AdEMAMix log: {ademamix_log}")


if __name__ == "__main__":
    main()
