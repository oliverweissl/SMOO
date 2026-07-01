#!/bin/bash
for sut in "default" "eff" "vit"; do
  for i in 1 2 3 4 5 6 7 8 850 963; do
    python experiments/test_hynea_script.py -c $i -n 10 -g 25 -o custom --gpu 1 --sut $sut
  done
done
