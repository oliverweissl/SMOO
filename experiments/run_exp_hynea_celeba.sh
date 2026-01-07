#!/bin/bash

for i in 1 5 7 13 15 16 18 20 31 35 ; do
  python examples/_hynea/test_hynea_script.py -c $i -n 10 -g 25 -o custom_binary --gpu 0 --lr 0.00001 --max_lr 0.0004
done