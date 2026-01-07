#!/bin/bash

for i in 1 2 3 4 5 6 7 8 850 963; do
    python examples/_hynea/test_mimicry_script.py -c $i -n 5 -g 25 -p 100 -o custom --gpu 0 &
    python examples/_hynea/test_mimicry_script.py -c $i -n 5 -g 25 -p 100 -o custom --gpu 1 &
    wait
done