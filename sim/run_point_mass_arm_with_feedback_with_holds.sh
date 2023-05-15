#!/bin/bash

N_SEEDS=5

for (( i=1; i<=$N_SEEDS; i++ ))
  do
    caffeinate python sim_point_mass_arm_with_feedback_with_holds.py --seed $i
  done