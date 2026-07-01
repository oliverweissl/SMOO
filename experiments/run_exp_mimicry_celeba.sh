#!/bin/bash

for i in 1 5 6 7 13 15 16 18 20 31 35; do
    python experiments/test_mimicry_script.py -c $i -n 10 -g 25 -p 100 -o custom_binary --gpu 1
done