import argparse
import os
import random
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import psutil


def init_process(rank, size, fn, master_port, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def p2p_sync(rank, size):
    if size == 1:
        return
    
    token = torch.tensor([1], dtype=torch.int64)

    if rank == 0:
        dist.send(token, dst=1)
        dist.recv(token, src=size - 1)
    else:
        dist.recv(token, src=rank - 1)
        dist.send(token, dst=(rank + 1) % size)


def reduce_scalar_max_to_rank0(value, rank, size):
    token = torch.tensor([float(value)], dtype=torch.float64)

    if size == 1:
        return float(token.item())
    
    if rank == 0:
        mx = float(token.item())
        buffer = torch.empty_like(token)
        for src in range(1, size):
            dist.recv(buffer, src=src)
            mx = max(mx, float(buffer.item()))
        return mx
    else:
        dist.send(token, dst=0)
        return 0.0


def reduce_scalar_mean_to_rank0(value, rank, size):
    token = torch.tensor([float(value)], dtype=torch.float64)

    if size == 1:
        return float(token.item())
    
    if rank == 0:
        sm = float(token.item())
        buffer = torch.empty_like(token)
        for src in range(1, size):
            dist.recv(buffer, src=src)
            sm += float(buffer.item())
        return sm / float(size)
    else:
        dist.send(token, dst=0)
        return 0.0


def rss_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def butterfly_allreduce(send, rank, size):
    """
    Performs Butterfly All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """

    buffer_for_chunk = torch.empty((size,), dtype=torch.float)

    send_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            send_futures.append(dist.isend(elem, i))

    recv_futures = []

    for i, elem in enumerate(buffer_for_chunk):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))
        else:
            elem.copy_(send[i])

    for future in recv_futures:
        future.wait()

    # compute the average
    torch.mean(buffer_for_chunk, dim=0, out=send[rank])

    for i in range(size):
        if i != rank:
            send_futures.append(dist.isend(send[rank], i))

    recv_futures = []

    for i, elem in enumerate(send):
        if i != rank:
            recv_futures.append(dist.irecv(elem, i))

    for future in recv_futures:
        future.wait()
    for future in send_futures:
        future.wait()


def ring_allreduce(send, rank, size):
    """
    Performs Ring All-Reduce over the process group. Modifies the input tensor in place.
    Args:
        send: torch.Tensor to be averaged with other processes.
        rank: Current process rank (in a range from 0 to size)
        size: Number of workers
    """
    n = send.numel()

    if size == 1 or n == 0:
        return
    
    prev_rank = (rank - 1) % size
    next_rank = (rank + 1) % size

    base = n // size
    remainder = n % size

    chunk_sizes = [base + (1 if r < remainder else 0) for r in range(size)]
    offsets = [0]
    for r in range(size):
        offsets.append(offsets[-1] + chunk_sizes[r])
    

    def get_chunk(tensor, idx):
        return tensor[offsets[idx] : offsets[idx + 1]]

    
    chunks = [get_chunk(send, idx).clone() for idx in range(size)]

    for step in range(size - 1):
        send_idx = (rank - step) % size
        recv_idx = (rank - step - 1) % size

        send_buffer = chunks[send_idx]
        recv_buffer = torch.empty(
            (chunk_sizes[recv_idx],),
            dtype=send.dtype,
            device=send.device,
        )

        send_future = dist.isend(send_buffer, next_rank)
        recv_future = dist.irecv(recv_buffer, prev_rank)

        recv_future.wait()
        send_future.wait()

        chunks[recv_idx] += recv_buffer
    
    owned_idx = (rank - (size - 1)) % size
    reduced_chunk = chunks[owned_idx]

    result_chunks = [None] * size
    result_chunks[owned_idx] = reduced_chunk

    for step in range(size - 1):
        send_idx = (owned_idx - step) % size
        recv_idx = (owned_idx - step - 1) % size

        send_buffer = result_chunks[send_idx]
        recv_buffer = torch.empty(
            (chunk_sizes[recv_idx],),
            dtype=send.dtype,
            device=send.device,
        )

        send_future = dist.isend(send_buffer, next_rank)
        recv_future = dist.irecv(recv_buffer, prev_rank)

        recv_future.wait()
        send_future.wait()

        result_chunks[recv_idx] = recv_buffer
    

    for r in range(size):
        get_chunk(send, r).copy_(result_chunks[r] / float(size))


def torch_allreduce(send, rank, size):
    dist.all_reduce(send, op=dist.ReduceOp.SUM)
    send /= float(size)


def make_tensor(rank, numel, dtype):
    torch.manual_seed(777 + rank)
    return torch.randn((numel,), dtype=dtype)


def run_one_bench(
    rank,
    size,
    impl,
    numel,
    iters,
    warmup,
    dtype,
    check,
):
    if impl == "ring":
        fn = ring_allreduce
    elif impl == "butterfly":
        fn = butterfly_allreduce
    elif impl == "torch":
        fn = torch_allreduce
    else:
        raise ValueError(f"unknown impl={impl}")
    
    ref0 = None
    if check:
        x0 = make_tensor(rank, numel, dtype).clone()
        ref0 = x0.clone()
        dist.all_reduce(ref0, op=dist.ReduceOp.SUM)
        ref0 /= float(size)

    rss0 = rss_mb()

    for _ in range(warmup):
        x = make_tensor(rank, numel, dtype)
        p2p_sync(rank, size)
        fn(x, rank, size)
    
    times_ms = []
    for _ in range(iters):
        x = make_tensor(rank, numel, dtype)
        p2p_sync(rank, size)
        t0 = time.perf_counter()
        fn(x, rank, size)
        p2p_sync(rank, size)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    
    rss1 = rss_mb()

    avg_ms_local = sum(times_ms) / len(times_ms)
    max_ms_local = max(times_ms)

    avg_ms = reduce_scalar_mean_to_rank0(avg_ms_local, rank, size)
    max_ms = reduce_scalar_max_to_rank0(max_ms_local, rank, size)

    max_abs_error = 0.0
    if check:
        x = make_tensor(rank, numel, dtype).clone()
        fn(x, rank, size)
        error_local = float((x - ref0).abs().max().item())
        max_abs_error = reduce_scalar_max_to_rank0(error_local, rank, size)
    
    if rank == 0:
        print(
            f"[impl={impl:9s}] world={size:2d} numel={numel:6d} "
            f"avg_ms={avg_ms:8.3f} max_ms={max_ms:8.3f} "
            f"rss_mb={rss1:8.1f} (+{(rss1 - rss0):6.1f}) "
            f"max_abs_error={max_abs_error:.3e}"
        )


def run_worker(rank, size, args):
    dtype = torch.float32 if args.dtype == "fp32" else torch.float64

    if args.sweep:
        for numel in args.numels:
            run_one_bench(
                rank=rank,
                size=size,
                impl=args.impl,
                numel=numel,
                iters=args.iters,
                warmup=args.warmup,
                dtype=dtype,
                check=args.check,
            )
    else:
        run_one_bench(
            rank=rank,
            size=size,
            impl=args.impl,
            numel=args.numel,
            iters=args.iters,
            warmup=args.warmup,
            dtype=dtype,
            check=args.check,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", type=str, default="ring", choices=["ring", "butterfly", "torch"])
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--numel", type=int, default=10000)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp64"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--master_port", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--numels",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000, 50000, 100000],
        help="used only with --sweep",
    )
    args = parser.parse_args()

    if args.impl == "butterfly" and (not args.sweep) and args.numel != args.world_size:
        raise SystemExit("butterfly_allreduce from template expects --numel == --world_size")
    
    if args.impl == "butterfly" and args.sweep:
        for x in args.numels:
            if x != args.world_size:
                raise SystemExit("butterfly_allreduce from template expects each numel == world_size in sweep")
    
    port = args.master_port if args.master_port != 0 else random.randint(25000, 30000)

    processes = []
    for rank in range(args.world_size):
        p = Process(target=init_process, args=(rank, args.world_size, lambda r, s: run_worker(r, s, args), port, args.backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
