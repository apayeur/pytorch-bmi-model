#!/bin/bash

N_SEEDS=10

# shellcheck disable=SC2004
for (( i=1; i<=$N_SEEDS; i++ ))
  do
    caffeinate python sim_point_mass_arm.py --seed $i
  done