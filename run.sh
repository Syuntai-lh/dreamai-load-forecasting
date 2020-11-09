#!/usr/bin/env bash

for method in dnn
do
  for i in $(seq 95 1 199)
  do
    arg1="--save_file=$i""_""$method""_3"
    arg2="--col_idx=$i"
    arg3="--method=$method"
    python tune-model.py "$arg1" "$arg2" "$arg3" --max_evals=100
    done
done