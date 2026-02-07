from typing import Optional
from collections import defaultdict, deque
import os
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer


MAX_LENGTH = 640


def _read_text_lines(data_path: str) -> list[str]:
    assert os.path.exists(data_path), f"File not found: {data_path}"
    with open(data_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file]
    return [line for line in lines if line]


def _shift_for_lm(ids: list[int]) -> tuple[list[int], list[int]]:
    """
    Build (input_ids, targets) for next token prediction.
    """
    if len(ids) < 2:
        return [], []
    return ids[:-1], ids[1:]


def _tokenize_to_ids(tokenizer, text: str) -> list[int]:
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return []
    return tokenizer.convert_tokens_to_ids(tokens)


class BrainDataset(Dataset):
    """
    Padding every sample to fixed max_length.
    """
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH) -> None:
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._pad_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0

        self._token_ids: list[list[int]] = []
        for text in _read_text_lines(data_path):
            ids = _tokenize_to_ids(self._tokenizer, text)
            if len(ids) >= 2:
                self._token_ids.append(ids)


    def __len__(self):
        return len(self._token_ids)


    def __getitem__(self, idx: int):
        ids = self._token_ids[idx][:self._max_length + 1]
        input_ids, targets = _shift_for_lm(ids)

        pad_size = self._max_length - len(input_ids)
        if pad_size > 0:
            input_ids = input_ids + [self._pad_id] * pad_size
            targets = targets + [self._pad_id] * pad_size
        
        return (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64)
        )
    

    @property
    def pad_id(self) -> int:
        return self._pad_id


class BigBrainDataset(BrainDataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH) -> None:
        super().__init__(data_path, max_length)

    def __getitem__(self, idx: int):
        ids = self._token_ids[idx][:self._max_length + 1]
        input_ids, targets = _shift_for_lm(ids)
        return (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64)
        )


class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH) -> None:
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._pad_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0

        self._token_ids: list[list[int]] = []
        self._len2idx: dict[int, list[int]] = defaultdict(list)

        for text in _read_text_lines(data_path):
            ids = _tokenize_to_ids(self._tokenizer, text)
            ids = ids[:self._max_length + 1]
            input_ids, _ = _shift_for_lm(ids)
            if len(input_ids) == 0:
                continue
            idx = len(self._token_ids)
            self._token_ids.append(ids)
            self._len2idx[len(input_ids)].append(idx)
            

    def __len__(self):
        return len(self._token_ids)
    

    def __getitem__(self, idx: int):
        ids = self._token_ids[idx]
        input_ids, targets = _shift_for_lm(ids)
        return (
            torch.tensor(input_ids, dtype=torch.int64),
            torch.tensor(targets, dtype=torch.int64)
        )
        
    
    @property
    def pad_id(self) -> int:
        return self._pad_id
    

    @property
    def len2idx(self) -> dict[int, list[int]]:
        return self._len2idx
    

    @property
    def max_length(self) -> int:
        return self._max_length
    


class UltraDuperBigBrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_length: int = MAX_LENGTH,
        algo: str = "basic",
    ) -> None:
        self._max_length = max_length
        self._packed_length = self._max_length + 1
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self._pad_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else 0
        self._algo = algo.lower()

        sequences: list[list[int]] = []
        for text in _read_text_lines(data_path):
            ids = _tokenize_to_ids(self._tokenizer, text)
            if len(ids) < 2:
                continue
            sequences.append(ids)
        
        if self._algo == "basic":
            sequences = [seq[:self._packed_length] for seq in sequences]
            self._packed = self._basic_pack(sequences)
        elif self._algo == "ffd":
            sequences = [seq for seq in sequences if len(seq) <= self._packed_length]
            self._packed = self._ffd_pack(sequences)
        elif self._algo == "obfd":
            sequences = [seq for seq in sequences if len(seq) <= self._packed_length]
            self._packed = self._obfd_pack(sequences)
        else:
            raise ValueError(f"Unknown algo={algo}. Use basic, ffd or obfd")


    def __len__(self):
        return len(self._packed)
    

    def __getitem__(self, idx: int):
        packed_ids, segment_ids = self._packed[idx]

        input_ids = packed_ids[:-1]
        targets_ids = packed_ids[1:]
        
        input_segments = segment_ids[:-1]
        target_segments = segment_ids[1:]

        is_boundary = (
            (input_segments != target_segments) |
            (input_segments == -1) |
            (target_segments == -1)
        )

        targets_ids = targets_ids.clone()
        targets_ids[is_boundary] = self._pad_id

        attention_mask = self._build_attention_mask(input_segments)

        return input_ids, targets_ids, attention_mask


    def _basic_pack(self, sequences: list[list[int]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        packed: list[tuple[torch.Tensor, torch.Tensor]] = []
        current_tokens: list[int] = []
        current_segments: list[int] = []
        current_segment_id = 0

        for seq in sequences:
            if len(current_tokens) + len(seq) > self._packed_length:
                pad_size = self._packed_length - len(current_tokens)
                if pad_size > 0:
                    current_tokens.extend([self.pad_id] * pad_size)
                    current_segments.extend([-1] * pad_size)
                
                packed.append((
                    torch.tensor(current_tokens, dtype=torch.int64),
                    torch.tensor(current_segments, dtype=torch.int64)
                ))

                current_tokens = []
                current_segments = []
                current_segment_id = 0
            
            free_space = self._packed_length - len(current_tokens)
            tokens_to_take = min(free_space, len(seq))
            
            current_tokens.extend(seq[:tokens_to_take])
            current_segments.extend([current_segment_id] * tokens_to_take)
            current_segment_id += 1

            if len(current_tokens) == self._packed_length:
                packed.append((
                    torch.tensor(current_tokens, dtype=torch.int64),
                    torch.tensor(current_segments, dtype=torch.int64)
                ))
                current_tokens = []
                current_segments = []
                current_segment_id = 0
        
        if current_tokens:
            pad_size = self._packed_length - len(current_tokens)
            if pad_size > 0:
                current_tokens.extend([self.pad_id] * pad_size)
                current_segments.extend([-1] * pad_size)
            
            packed.append((
                torch.tensor(current_tokens, dtype=torch.int64),
                torch.tensor(current_segments, dtype=torch.int64)
            ))

        return packed
    

    def _ffd_pack(self, sequences: list[list[int]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        sequences = sorted(sequences, key=len, reverse=True)
        bins: list[list[list[int]]] = []
        remaining_space: list[int] = []

        for seq in sequences:
            seq_length = len(seq)

            for bin_index, free_space in enumerate(remaining_space):
                if seq_length <= free_space:
                    bins[bin_index].append(seq)
                    remaining_space[bin_index] -= seq_length
                    break
            else:
                bins.append([seq])
                remaining_space.append(self._packed_length - seq_length)
        
        return self._materialize_bins(bins)
    

    def _obfd_pack(self, sequences: list[list[int]]) -> list[tuple[torch.Tensor, torch.Tensor]]:

        def next_pow2(x: int) -> int:
            p = 1
            while p < x:
                p <<= 1
            return p
        
        sequences = sorted(sequences, key=len, reverse=True)
        
        bin_capacity = self._packed_length
        
        bins: list[list[list[int]]] = []
        remaining_space: list[int] = []
        bins_by_free_space: list[list[int]] = [[] for _ in range(bin_capacity + 1)]
        segment_tree: list[int] = [0] * (2 * next_pow2(bin_capacity + 1))
        tree_base = len(segment_tree) // 2

        def tree_set(pos: int, value: int) -> None:
            i = tree_base + pos
            segment_tree[i] = value
            i >>= 1
            while i:
                segment_tree[i] = segment_tree[2 * i] | segment_tree[2 * i + 1]
                i >>= 1


        def has_free_space_in_range(left: int, right: int) -> bool:
            if left > right:
                return False

            left += tree_base
            right += tree_base
            found = 0
            
            while left <= right:
                if left & 1:
                    found |= segment_tree[left]
                    left += 1
                if not (right & 1):
                    found |= segment_tree[right]
                    right -= 1
                left >>= 1
                right >>= 1

            return found == 1
        

        def find_best_remaining_space(needed: int) -> Optional[int]:
            if needed > bin_capacity:
                return None
            
            if not has_free_space_in_range(needed, bin_capacity):
                return None
            
            node = 1
            seg_left, seg_right = 0, tree_base - 1

            while seg_left != seg_right:
                mid = seg_left + ((seg_right - seg_left) >> 1)

                left_node = 2 * node
                right_node = left_node + 1

                left_l, left_r = seg_left, mid
                if needed <= left_r and segment_tree[left_node] == 1:
                    left = max(needed, left_l)
                    right = min(bin_capacity, left_r)
                    if left <= right and has_free_space_in_range(left, right):
                        node = left_node
                        seg_right = mid
                        continue
                
                node = right_node
                seg_left = mid + 1

            if seg_left > bin_capacity:
                return None
            
            return seg_left

        
        for seq in sequences:
            seq_length = len(seq)
            best_space = find_best_remaining_space(seq_length)

            if best_space is None:
                bin_index = len(bins)
                bins.append([seq])
                free = bin_capacity - seq_length
                remaining_space.append(free)
                bins_by_free_space[free].append(bin_index)
                tree_set(free, 1)
            else:
                bin_index = bins_by_free_space[best_space].pop()
                if not bins_by_free_space[best_space]:
                    tree_set(best_space, 0)
                bins[bin_index].append(seq)
                new_free = best_space - seq_length
                remaining_space[bin_index] = new_free
                bins_by_free_space[new_free].append(bin_index)
                tree_set(new_free, 1)
        
        return self._materialize_bins(bins)


    @property
    def pad_id(self) -> int:
        return self._pad_id
    

    def _materialize_bins(self, bins: list[list[list[int]]]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        packed: list[tuple[torch.Tensor, torch.Tensor]] = []
        for seq_list in bins:
            tokens: list[int] = []
            segments: list[int] = []
            segment_id = 0

            for seq in seq_list:
                if len(tokens) + len(seq) > self._packed_length:
                    break
                tokens.extend(seq)
                segments.extend([segment_id] * len(seq))
                segment_id += 1
            pad_size = self._packed_length - len(tokens)
            if pad_size > 0:
                tokens.extend([self._pad_id] * pad_size)
                segments.extend([-1] * pad_size)
            packed.append((
                torch.tensor(tokens, dtype=torch.int64),
                torch.tensor(segments, dtype=torch.int64)
            ))
        
        return packed
    

    def _build_attention_mask(self, segment_ids: torch.Tensor) -> torch.Tensor:
        seq_length = segment_ids.size(0)
        device = segment_ids.device

        row = torch.arange(seq_length, device=device).view(seq_length, 1)
        col = torch.arange(seq_length, device=device).view(1, seq_length)

        causal_mask = col <= row

        segment_mask = (
            (segment_ids.view(seq_length, 1) == segment_ids.view(1, seq_length)) &
            (segment_ids.view(seq_length, 1) != -1)
        )

        valid_positions = causal_mask & segment_mask

        mask = torch.full((seq_length, seq_length), float("-inf"), device=device)
        mask[valid_positions] = 0.0

        return mask


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    max_length: Optional[int] = MAX_LENGTH,
    pad_id: int = 0
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param max_length: maximum sequence length to pad to (for "Brain" approach only)
    :return: tuple of padded sequences and corresponding training targets
    """
    has_mask = len(batch[0]) == 3

    if has_mask:
        input_sequences, target_sequences, masks = zip(*batch)
    else:
        input_sequences, target_sequences = zip(*batch)
        masks = None

    lengths = [seq.size(0) for seq in input_sequences]
    sequence_length = max(lengths) if max_length is None else int(max_length)
    batch_size = len(input_sequences)

    padded_inputs = torch.full((batch_size, sequence_length), fill_value=pad_id, dtype=torch.int64)
    padded_targets = torch.full((batch_size, sequence_length), fill_value=pad_id, dtype=torch.int64)

    for i, (input_seq, target_seq) in enumerate(zip(input_sequences, target_sequences)):
        length = min(sequence_length, input_seq.size(0))
        padded_inputs[i, :length] = input_seq[:length]
        padded_targets[i, :length] = target_seq[:length]

    if not has_mask:
        return padded_inputs, padded_targets

    padded_masks = torch.full(
        (batch_size, sequence_length, sequence_length),
        fill_value=float("-inf"),
        dtype=torch.float32,
    )

    for i, mask in enumerate(masks):
        length = min(sequence_length, mask.size(0))
        padded_masks[i, :length, :length] = mask[:length, :length].to(torch.float32)

    return padded_inputs, padded_targets, padded_masks



class UltraBigBrainBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: UltraBigBrainDataset,
        batch_size: int,
        k: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._k = k
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = random.Random(seed)

        self._buckets: dict[int, list[int]] = {}
        for length, indices in self._dataset.len2idx.items():
            lst = indices.copy()
            if self._shuffle:
                self._rng.shuffle(lst)
            self._buckets[length] = lst
        
        self._lengths: list[int] = sorted(self._buckets.keys())
        self._pos: dict[int, int] = {length: 0 for length in self._lengths}


    def __len__(self):
        size = len(self._dataset)
        return size // self._batch_size if self._drop_last else (size + self._batch_size - 1) // self._batch_size
    

    def __iter__(self):
        for length in self._lengths:
            self._pos[length] = 0
        
        active = self._lengths[:]

        while True:
            active = [length for length in active if self._pos[length] < len(self._buckets[length])]
            if not active:
                break

            base_length = self._rng.choice(active) if self._shuffle else active[0]

            batch: list[int] = []
            max_length = min(base_length + self._k, self._dataset.max_length)

            for length in range(base_length, max_length + 1):
                indices = self._buckets.get(length)
                if not indices:
                    continue

                idx = self._pos.get(length, 0)
                while idx < len(indices) and len(batch) < self._batch_size:
                    batch.append(indices[idx])
                    idx += 1
                self._pos[length] = idx

                if len(batch) == self._batch_size:
                    break

            if self._drop_last and len(batch) < self._batch_size:
                continue
            
            if batch:
                yield batch
