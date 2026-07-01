#!/bin/bash
for sut in "default" "eff" "vit"; do
  for i in 1 2 3 4 5 6 7 8 850 963; do
      python experiments/test_mimicry_script.py -c $i -n 5 -g 25 -p 100 -o custom --gpu 0 --sut $sut &
      python experiments/test_mimicry_script.py -c $i -n 5 -g 25 -p 100 -o custom --gpu 1 --sut $sut &
      wait
  done
done