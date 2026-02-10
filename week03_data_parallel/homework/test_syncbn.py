import torch
from syncbn import SyncBatchNorm

import pytest
import torch.distributed as dist
import os
import random
import datetime


def _init_process_group(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=30))


def _worker_fn(rank, world_size, port, x_rank, hid_dim, batch_size, q):
    _init_process_group(rank, world_size, port)

    x_rank = x_rank.clone().detach().requires_grad_(True)

    batch_norm = SyncBatchNorm(hid_dim)
    batch_norm.train()
    y_rank = batch_norm(x_rank)

    total_batch = world_size * batch_size
    half_batch = total_batch // 2
    start = rank * batch_size
    end = start + batch_size
    left = max(start, 0)
    right = min(end, half_batch)

    loss = y_rank.new_tensor(0.0)
    if left < right:
        loss = y_rank[(left - start):(right - start)].sum()
    
    loss.backward()

    q.put((rank, y_rank.detach().cpu().numpy(), x_rank.grad.detach().cpu().numpy()))

    dist.destroy_process_group()


@pytest.mark.parametrize("num_workers", [1, 4])
@pytest.mark.parametrize("hid_dim", [128, 256, 512, 1024])
@pytest.mark.parametrize("batch_size", [32, 64])
def test_batchnorm(num_workers, hid_dim, batch_size):
    torch.manual_seed(0)
    total_batch = num_workers * batch_size
    x_full = torch.randn((total_batch, hid_dim), dtype=torch.float32)

    x_ref = x_full.clone().detach().requires_grad_(True)
    batch_norm = torch.nn.BatchNorm1d(hid_dim, affine=False, track_running_stats=False)
    batch_norm.train()

    y_ref = batch_norm(x_ref)
    loss_ref = y_ref[:(total_batch // 2)].sum()
    loss_ref.backward()

    y_ref_full = y_ref.detach().cpu()
    dx_ref_full = x_ref.grad.detach().cpu()


    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()

    port = random.randint(25000, 40000)
    processes = []
    for rank in range(num_workers):
        x_rank = x_full[rank * batch_size : (rank + 1) * batch_size]
        p = ctx.Process(target=_worker_fn, args=(rank, num_workers, port, x_rank, hid_dim, batch_size, q))
        p.start()
        processes.append(p)
    

    outputs = [None] * num_workers
    grads = [None] * num_workers

    for _ in range(num_workers):
        rank, y_rank, dx_rank = q.get()
        outputs[rank] = torch.from_numpy(y_rank)
        grads[rank] = torch.from_numpy(dx_rank)
    
    for p in processes:
        p.join()
    
    y_sync_full = torch.cat(outputs, dim=0)
    dx_sync_full = torch.cat(grads, dim=0)

    torch.testing.assert_close(y_sync_full, y_ref_full, rtol=0.0, atol=1e-3)
    torch.testing.assert_close(dx_sync_full, dx_ref_full, rtol=0.0, atol=1e-3)
