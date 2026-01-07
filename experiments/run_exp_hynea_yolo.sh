#!/bin/bash

for i in 2 5 7 9 11; do
  python examples/_hynea/test_hynea_script.py -c $i -n 10 -g 25 -o custom_yolo --gpu 0 --lr 0.0001  --max_lr 0.004
done

wait