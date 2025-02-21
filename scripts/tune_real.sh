#!/bin/bash


trap 'for pid in "${main_pids[@]}"; do kill $pid; done; exit' INT
main_pids=()
# Array of values for tune-wds
tune_wds_values=("0.0" "1e-1" "1e-2" "1e-3" "1e-4")
current_device=$1
# Loop through the values and run the script in the background
for wds in "${tune_wds_values[@]}"; do
    CUDA_VISIBLE_DEVICES=$current_device python scripts/tune_hyp.py \
        --exp-config $2 \
        --epoch $3 $3 \
        --max-depth 10 \
        --max-syn-depth 1 \
        --modalities $4 $5 \
        --tune-lrs 1e-3 5e-4 1e-4 5e-5 1e-5 \
        --tune-wds "$wds" \
        --task $6 \
        --eval-task $7 \
        --tune-only "all" \
        --use-saved-model &
    main_pids+=($!)
    # Increment the device number and mod by 8
    current_device=$(( (current_device + 1) % 8 ))
done

echo ${main_pids[@]}
# Wait for all background processes to finish
wait

tune_wds_values_string=$(IFS=' '; echo "${tune_wds_values[*]}")
CUDA_VISIBLE_DEVICES=$1 python scripts/tune_hyp.py \
        --exp-config $2 \
        --epoch $3 $3 \
        --max-depth 10 \
        --max-syn-depth 1 \
        --modalities $4 $5 \
        --tune-lrs 1e-3 5e-4 1e-4 5e-5 1e-5 \
        --tune-wds $tune_wds_values_string \
        --task $6 \
        --eval-task $7 \
        --base-seed $8 \
        --use-saved-model 