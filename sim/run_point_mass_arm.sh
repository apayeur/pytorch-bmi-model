#!/bin/bash

N_SEEDS=1

for (( i=1; i<=$N_SEEDS; i++ ))
  do
    caffeinate python sim_point_mass_arm.py --seed $i
  done