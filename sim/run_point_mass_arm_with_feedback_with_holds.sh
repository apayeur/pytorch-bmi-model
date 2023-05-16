#!/bin/bash

N_SEEDS=10

for (( i=6; i<=$N_SEEDS; i++ ))
  do
    caffeinate python sim_point_mass_arm_with_feedback_with_holds.py --seed $i
  done