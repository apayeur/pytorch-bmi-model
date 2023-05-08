#!/bin/bash

N_SEEDS=10

for (( i=1; i<=$N_SEEDS; i++ ))
  do
    for clda in 0. 0.1 0.5
    do
      caffeinate python sim_clda.py --seed $i --clda $clda
    done
  done