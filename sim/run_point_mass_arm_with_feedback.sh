#!/bin/bash

N_SEEDS=10

for (( i=1; i<=$N_SEEDS; i++ ))
  do
    caffeinate python sim_point_mass_arm_with_feedback.py --seed $i
  done