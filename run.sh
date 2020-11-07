#!/usr/bin/env bash

for method in dnn svr rf dct extra
do
  for i in $(seq 0 1 200)
  do
    arg1="--save_file=$i""_""$method"
    arg2="--col_idx=$i"
    arg3="--method=$method"
    python tune-model.py "$arg1" "$arg2" "$arg3" --max_evals=100
    done
done