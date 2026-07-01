#!/bin/bash

pairs=(
  "5e-6 8e-5"
  "1e-5 4e-4"
  "5e-5 8e-4"
  "1e-4 4e-3" ## MIddle
  "5e-4 8e-3"
  "1e-3 4e-2"
  "5e-3 8e-2"
  "1e-2 4e-1"
  "5e-2 8e-1"
)

for i in 2 5 7 9 11; do
  for pair in "${pairs[@]}"; do
    lr=$(echo "$pair" | awk '{print $1}')
    max_lr=$(echo "$pair" | awk '{print $2}')

    python experiments/test_hynea_script.py -c $i -n 1 -g 25 -o custom_yolo --gpu 0 --lr $lr --max_lr $max_lr --path "${lr}_${max_lr}_" &
    pid0=$!

    python experiments/test_hynea_script.py -c $i -n 1 -g 25 -o custom_yolo --gpu 1 --lr $lr --max_lr $max_lr --path "${lr}_${max_lr}_" &
    pid1=$!

    wait "$pid0"
    wait "$pid1"
  done
done

wait