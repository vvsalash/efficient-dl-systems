import os
import argparse
import torch
import time
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR100

from syncbn import SyncBatchNorm

torch.set_num_threads(1)


def init_process(local_rank, fn, backend="nccl"):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)


class Net(nn.Module):
    """
    A very simple model with minimal changes from the tutorial, used for the sake of simplicity.
    Feel free to replace it with EffNetV2-XL once you get comfortable injecting SyncBN into models programmatically.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 100)
        self.bn1 = nn.BatchNorm1d(128, affine=False)  # to be replaced with SyncBatchNorm

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def replace_bn_with_syncbn(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d) and (child.affine is False):
            setattr(module, name, SyncBatchNorm(child.num_features, child.eps, child.momentum))
        else:
            replace_bn_with_syncbn(child)


@torch.no_grad()
def distributed_mean_scalar(x: torch.Tensor):
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= dist.get_world_size()
    return x


def run_training(rank, size, args):
    torch.manual_seed(0)

    if args.device == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    is_rank0 = (rank == 0)

    dataset = CIFAR100(
        "./cifar",
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]
        ),
        download=True,
    )
    n_val = int(len(dataset) * args.val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(0)
    )


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(
            train_ds,
            num_replicas=size,
            rank=rank,
            shuffle=True,
            drop_last=True
        ),
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(
            val_ds,
            num_replicas=size,
            rank=rank,
            shuffle=False,
            drop_last=False
        ),
        num_workers=2,
        pin_memory=True
    )

    model = Net()

    if args.impl == "custom":
        replace_bn_with_syncbn(model)

    model.to(device)

    if args.impl == "torch":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank] if device.type == "cuda" else None,
            output_device=rank if device.type == "cuda" else None,
        )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        t0 = time.time()
        epoch_loss = 0.0
        epoch_acc = 0.0
        steps = 0

        optimizer.zero_grad(set_to_none=True)

        for i, (data, target) in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            do_step = ((i + 1) % args.grad_accum_steps == 0)

            ddp_no_sync = (args.impl == "torch" and not do_step)
            ctx = model.no_sync() if ddp_no_sync else torch.enable_grad()

            with ctx:
                output = model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                (loss / args.grad_accum_steps).backward()
            
            if do_step:
                if args.impl == "custom":
                    average_gradients(model)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                acc = (output.argmax(dim=1) == target).float().mean()
                epoch_loss += loss.detach().item()
                epoch_acc += acc.detach().item()
                steps += 1
            
            if is_rank0 and (i % args.log_every == 0):
                print(
                    f"epoch={epoch} iter={i} "
                    f"loss={epoch_loss/steps:.4f} acc={epoch_acc/steps:.4f} "
                    f"do_step={do_step}"
                )
            
        
        loss_t = torch.tensor([epoch_loss / steps], device=device)
        acc_t = torch.tensor([epoch_acc / steps], device=device)

        distributed_mean_scalar(loss_t)
        distributed_mean_scalar(acc_t)

        model.eval()

        val_correct = torch.tensor([0.0], device=device)
        val_total = torch.tensor([0.0], device=device)

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data)
                pred = output.argmax(dim=1)
                val_correct += (pred == target).float().sum()
                val_total += torch.tensor([target.numel()], device=device, dtype=torch.float32)
        
        dist.all_reduce(val_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_total, op=dist.ReduceOp.SUM)
        val_acc = (val_correct / val_total).item()

        dt = time.time() - t0
        peak_memory = None
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)

        if is_rank0:
            message = f"[epoch {epoch}] train_loss={loss_t.item():.4f} train_acc={acc_t.item():.4f} val_acc={val_acc:.4f} time={dt:.2f}s"
            if peak_memory is not None:
                message += f" peak_memory={peak_memory:.1f}MB"
            print(message)
            

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--impl", choices=["custom", "torch"], default="custom")
    p.add_argument("--device", default="cpu")
    p.add_argument("--backend", default="gloo")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--grad_accum_steps", type=int, default=2)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.1)
    args = p.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(
        local_rank,
        fn=lambda rank, size: run_training(rank, size, args),
        backend=args.backend
    )
