#!/bin/bash

pairs=(
  "1e-8 1e-6"
  "3e-8 3e-6"
  "1e-7 1e-5"
  "3e-7 3e-5"
  "1e-6 1e-4"
  "3e-6 3e-4"
  "1e-5 1e-3"
  "3e-5 3e-3"
  "1e-4 1e-2"
)

for i in 1 2 3 4 5 6 7 8 850 963; do
  for pair in "${pairs[@]}"; do
    lr=$(echo $pair | awk '{print $1}')
    max_lr=$(echo $pair | awk '{print $2}')

    python examples/_hynea/test_hynea_script.py -c $i -n 1 -g 25 -o custom --gpu 0 --lr $lr --max_lr $max_lr --path "${lr}_${max_lr}_" &
    pid0=$!

    python examples/_hynea/test_hynea_script.py -c $i -n 1 -g 25 -o custom --gpu 1 --lr $lr --max_lr $max_lr --path "${lr}_${max_lr}_" &
    pid1=$!

    wait "$pid0"
    wait "$pid1"
  done
done