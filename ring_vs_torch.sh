#!/bin/bash

for w in 1 2 4 8 16; do
  python allreduce.py --impl ring  --world_size $w --sweep --check
  python allreduce.py --impl torch --world_size $w --sweep --check
done

