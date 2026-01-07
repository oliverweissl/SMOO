#!/bin/bash

pairs=(
  "1e-7 4e-6"
  "3e-7 1e-5"
  "1e-6 4e-5"
  "3e-6 1e-4"
  "1e-5 4e-4"
  "3e-5 1e-3"
  "1e-4 4e-3"
  "3e-4 1e-2"
)

for i in 1 5 7 13 15 16 18 20 31 35; do
  for pair in "${pairs[@]}"; do
    lr=$(echo $pair | awk '{print $1}')
    max_lr=$(echo $pair | awk '{print $2}')

    python examples/_hynea/test_hynea_script.py -c $i -n 1 -g 25 -o custom_binary --gpu 0 --lr $lr --max_lr $max_lr --path "${lr}_${max_lr}_" &
    pid0=$!

    python examples/_hynea/test_hynea_script.py -c $i -n 1 -g 25 -o custom_binary --gpu 1 --lr $lr --max_lr $max_lr --path "${lr}_${max_lr}_" &
    pid1=$!

    wait "$pid0"
    wait "$pid1"
  done
done