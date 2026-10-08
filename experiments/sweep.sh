#!/usr/bin/env bash
# Hyperparameter sweep: one training run per GPU, in parallel.
#
#   bash experiments/sweep.sh NAME "ARGS_GPU0" "ARGS_GPU1" ...
#
# Run k gets CUDA_VISIBLE_DEVICES=k, writes to outputs/train/NAME_k/ and logs
# to outputs/train/NAME_k.txt (the tqdm bar refreshes every 30 s there).
# COMMON_ARGS is appended to every run.  Example (lambda sweep):
#
#   COMMON_ARGS="--batch 8 --steps 2000 --val-every 250" \
#   bash experiments/sweep.sh lam "--lam 3e-4" "--lam 1e-3" "--lam 3e-3" "--lam 1e-2"
#
# Uses $PYTHON (default: python).  Waits for all runs and prints their final
# validation records.

set -u
name=$1
shift
py=${PYTHON:-python}
k=0
pids=()
runs=()
for args in "$@"; do
    out=outputs/train/${name}_${k}
    mkdir -p outputs/train
    echo "GPU $k: $args -> $out"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$k nohup "$py" -m experiments.train_lpd --out "$out" $args ${COMMON_ARGS:-} \
        > "$out.txt" 2>&1 &
    pids+=($!)
    runs+=("$args")
    k=$((k + 1))
done
wait "${pids[@]}"
for ((i = 0; i < k; i++)); do
    echo "== ${name}_${i}: ${runs[$i]}"
    grep val_auc "outputs/train/${name}_${i}/log.jsonl" | tail -1
done
