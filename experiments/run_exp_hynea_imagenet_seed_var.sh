#!/bin/bash
for seed in 1 9127 23481 48673 938124; do
  for i in 1 2 3 4 5 6 7 8 850 963; do
    python experiments/test_hynea_script.py -c $i -n 10 -g 25 -o custom --gpu 1 --seed $seed
  done
done
