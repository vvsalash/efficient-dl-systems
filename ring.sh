#!/bin/bash

for w in 1 2 4 8 16 32; do
  python allreduce.py --impl ring --world_size $w --sweep --check
done

