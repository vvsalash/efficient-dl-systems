from enum import Enum
from dataclasses import dataclass
from typing import Optional
import time
import argparse

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore", message="huggingface/tokenizers")
warnings.filterwarnings("ignore", message="Token indices sequence length is longer")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from dataset import (
    MAX_LENGTH,
    BrainDataset,
    BigBrainDataset,
    UltraBigBrainDataset,
    UltraDuperBigBrainDataset,
    UltraBigBrainBatchSampler,
    collate_fn,
)
from transformer import PositionalEncoding, generate_square_subsequent_mask


@dataclass
class BenchRow:
    mode: str
    variant: str
    batch_size: int
    max_length: int
    min_ms: float
    max_ms: float
    mean_ms: float
    median_ms: float
    n_batches: int


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 1024,
        nhead: int = 8,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.nhead = nhead
        self._embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self._pos = PositionalEncoding(d_model, dropout=0.1, max_len=MAX_LENGTH + 5)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            batch_first=False,
        )

        self._decoder = nn.TransformerDecoder(layer, num_layers=1)
        self._lm_head = nn.Linear(d_model, vocab_size)

    
    def forward(self, input_ids: torch.Tensor, tgt_mask: torch.Tensor):
        input_ids = input_ids.t().contiguous()
        h = self._embedding(input_ids)
        h = self._pos(h)
        memory = torch.zeros(1, h.size(1), h.size(2), device=h.device, dtype=h.dtype)
        out = self._decoder(tgt=h, memory=memory, tgt_mask=tgt_mask)
        logits = self._lm_head(out)
        return logits


@torch.no_grad()
def _warmup_gpu():
    if torch.cuda.is_available():
        x = torch.randn(1024, 1024, device="cuda")
        for _ in range(10):
            x = x @ x
        torch.cuda.synchronize()


def _bench_loader(
    model: GPT2,
    data_loader: DataLoader,
    pad_id: int,
    device: torch.device,
    warmup_steps: int,
    max_batches: Optional[int],
    packed: bool,
    desc: str,
):
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    times = []
    batches_seen = 0

    total = max_batches if max_batches is not None else None

    for batch in tqdm(
        data_loader,
        total=total,
        desc=desc,
        leave=False,
    ):
        if packed:
            input_ids, targets, attn_mask = batch
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            
            attn_mask = attn_mask.repeat_interleave(model.nhead, dim=0)
            tgt_mask = attn_mask
        else:
            input_ids, targets = batch
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            length = input_ids.size(1)
            tgt_mask = generate_square_subsequent_mask(length).to(device)
        

        if batches_seen < warmup_steps:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, tgt_mask=tgt_mask)
            targets_t = targets.t().contiguous()
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets_t.reshape(-1))
            loss.backward()
            optimizer.step()
            batches_seen += 1
            continue

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, tgt_mask=tgt_mask)

        targets_t = targets.t().contiguous()
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets_t.reshape(-1))
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        t1 = time.perf_counter()

        times.append(t1 - t0)
        batches_seen += 1

        if max_batches is not None and len(times) >= max_batches:
            break

    if len(times) == 0:
        return None
    
    arr = np.array(times, dtype=np.float64) * 1000.0
    return float(arr.min()), float(arr.max()), float(arr.mean()), float(np.median(arr)), len(times)




def run_all(
    data_path: str,
    batch_size: int,
    device: torch.device,
    warmup_steps: int,
    max_batches: Optional[int]
) -> None:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    _warmup_gpu()

    rows: list[BenchRow] = []

    def add_row(
        mode: str,
        variant: str,
        packed: bool,
        loader: DataLoader,
    ):
        desc = f"{mode}-{variant}"
        model = GPT2(
            vocab_size=tokenizer.vocab_size,
            pad_id=pad_id
        ).to(device)

        stats = _bench_loader(
            model=model,
            data_loader=loader,
            pad_id=pad_id,
            device=device,
            warmup_steps=warmup_steps,
            max_batches=max_batches,
            packed=packed,
            desc=desc
        )

        if stats is None:
            return

        min_ms, max_ms, mean_ms, median_ms, n_batches = stats
        rows.append(
            BenchRow(
                mode=mode,
                variant=variant,
                batch_size=batch_size,
                max_length=MAX_LENGTH,
                min_ms=min_ms,
                max_ms=max_ms,
                mean_ms=mean_ms,
                median_ms=median_ms,
                n_batches=n_batches
            )
        )
    
    dataset = BrainDataset(data_path=data_path, max_length=MAX_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, max_length=MAX_LENGTH, pad_id=dataset.pad_id),
    )
    add_row("BRAIN", "max_length=640", packed=False, loader=loader)


    dataset = BigBrainDataset(data_path=data_path, max_length=MAX_LENGTH)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda batch: collate_fn(batch, max_length=None, pad_id=dataset.pad_id),
    )
    add_row("BIG_BRAIN", "pad_to_batch_max", packed=False, loader=loader)


    for k in [1, 5, 10, 20, 50, 640]:
        dataset = UltraBigBrainDataset(
            data_path=data_path,
            max_length=MAX_LENGTH
        )
        batch_sampler = UltraBigBrainBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            k=k,
            shuffle=True,
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(batch, max_length=None, pad_id=dataset.pad_id),
        )
        add_row("ULTRA_BIG_BRAIN", f"k={k}", packed=False, loader=loader)
    

    for algo in ["basic", "ffd", "obfd"]:
        dataset = UltraDuperBigBrainDataset(
            data_path=data_path,
            max_length=MAX_LENGTH,
            algo=algo,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda batch: collate_fn(batch, max_length=None, pad_id=dataset.pad_id),
        )
        add_row("ULTRA_DUPER_BIG_BRAIN", f"algo={algo}", packed=True, loader=loader)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.sort_values(["mode", "variant"]).reset_index(drop=True)
    df.to_csv("benchmark.csv", index=False)
    print(df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to WikiText-103 train text file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--max_batches", type=int, default=200, help="limit measured batches (None = full epoch)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_all(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        warmup_steps=args.warmup_steps,
        max_batches=args.max_batches if args.max_batches > 0 else None,
    )


if __name__ == "__main__":
    main()
