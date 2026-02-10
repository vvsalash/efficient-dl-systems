import argparse
import os
import time
import torch
import torch.distributed as dist

from syncbn import SyncBatchNorm


def init_process_group(backend: str):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local


def local_slice_for_first_half(rank: int, batch_size: int, world_size: int):
    total_batch = world_size * batch_size
    half = total_batch // 2
    start = rank * batch_size
    end = start + batch_size
    left = max(start, 0)
    right = min(end, half)
    if left >= right:
        return 0, 0
    return left - start, right - start


def make_bn(
    impl: str,
    num_features: int,
    eps: float,
    momentum: float,
    device
):
    if impl == "custom":
        bn = SyncBatchNorm(num_features, eps, momentum)
    else:
        bn = torch.nn.SyncBatchNorm(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=False,
            track_running_stats=True
        )
    return bn.to(device).train()


@torch.no_grad()
def distributed_max(x: torch.Tensor):
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    return x


def bench(
    impl: str,
    num_features: int,
    batch_size: int,
    eps: float,
    momentum: float,
    warmup: int,
    iters: int,
    device,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    bn = make_bn(impl, num_features, eps, momentum, device)

    torch.manual_seed(0)
    x = torch.randn(
        batch_size,
        num_features,
        device=device,
        dtype=torch.float32,
        requires_grad=True
    )
    left, right = local_slice_for_first_half(rank, batch_size, world_size)

    def step():
        if x.grad is not None:
            x.grad.zero_()
        y = bn(x)
        loss = y[left:right].sum() if left < right else y.sum() * 0.0
        loss.backward()
    
    dist.barrier()
    for _ in range(warmup):
        step()
    dist.barrier()

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            step()
        t1.record()
        torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / iters
        memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            step()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0 / iters
        memory = float("nan")
    

    ms = distributed_max(torch.tensor([ms], device=device)).item()
    memory = distributed_max(
        torch.tensor([memory], device=device)
    ).item() if device.type == "cuda" else memory
    dist.barrier()

    return ms, memory



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["custom", "torch"], required=True)
    p.add_argument("--backend", default="nccl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-5)
    p.add_argument("--momentum", type=float, default=0.1)
    args = p.parse_args()

    rank, world_size, local = init_process_group(args.backend)

    if args.device == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    
    hid_dims = [128, 256, 512, 1024]
    batch_sizes = [32, 64]

    if rank == 0:
        print(f"impl={args.impl} world_size={world_size} device={device}")
        print(f"{'hid':>6} {'bs':>4} {'ms/iter':>10} {'peakMB':>10}")

    
    for num_features in hid_dims:
        for batch_size in batch_sizes:
            ms, memory = bench(
                args.impl,
                num_features,
                batch_size,
                args.eps,
                args.momentum,
                args.warmup,
                args.iters,
                device
            )

            if rank == 0:
                if device.type == "cuda":
                    print(f"{num_features:6d} {batch_size:4d} {ms:10.3f} {memory:10.1f}")
                else:
                    print(f"{num_features:6d} {batch_size:4d} {ms:10.3f} {'N/A':10.1f}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()