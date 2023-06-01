#!/bin/bash

N_SEEDS=5

for (( i=1; i<=$N_SEEDS; i++ ))
  do
    for clda in 0. 0.1 0.2 0.5
    do
      echo -e "Simulating BCI task for seed $i, CLDA=$clda"
      caffeinate python sim_clda.py --seed $i --clda $clda
      echo -e "\n"
    done
  done