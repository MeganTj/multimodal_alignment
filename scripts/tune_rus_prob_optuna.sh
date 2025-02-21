#!/bin/bash

# Trap to kill background processes on interrupt
trap 'for pid in "${main_pids[@]}"; do kill $pid; done; exit' INT

# Function to execute the loop for a given sequence of indices
run_experiments() {
    local device_list=("${!1}")  # The list of CUDA devices passed as an array
    local exp_config=$2
    local dataset_name=$3
    local epoch_mod1=$4
    local epoch_mod2=$5
    local seed=$6
    local indices=("${@:7}")  # Indices are passed starting from the 6th argument

    main_pids=() # Reset the list of process IDs
    device_count=${#device_list[@]}  # Get the number of devices
    device_index=0  # Initialize the device index

    # Loop through each index in the sequence
    for i in "${indices[@]}"; do
        # Construct the full red_key value
        local red_key="feat_8_r_${i}"

        # Get the current device from the list
        current_device=${device_list[$device_index]}

        CUDA_VISIBLE_DEVICES=$current_device python scripts/tune_hyp_optuna.py \
            --exp-config "$exp_config" \
            --dataset-name "$dataset_name" \
            --red-key "$red_key" \
            --epoch "$epoch_mod1" "$epoch_mod2" \
            --max-depth 4 \
            --max-syn-depth 4 \
            --tune-lrs 1e-3 5e-3 1e-2 5e-2 \
            --tune-wds 0.0 1e-4 1e-3 1e-2 1e-1 \
            --tune-only "diag" \
            --base-seed "$seed" \
            --use-saved-model &

        main_pids+=($!) # Add the process ID to the list

        # Increment the device index and wrap around if necessary
        device_index=$(( (device_index + 1) % device_count ))
    done

    echo "${main_pids[@]}"
    wait # Wait for all processes to complete
}

# Read arguments
device_list_input=$1  # Second argument: Comma-separated list of devices
exp_config=$2      # Third argument: Experiment configuration
dataset_name=$3    # Fourth argument: Dataset name
epoch_mod1=$4      # Fifth argument: Epoch start
epoch_mod2=$5      # Sixth argument: Epoch end
seed=$6    # Seed

# Convert device list into an array
IFS=',' read -r -a device_list <<< "$device_list_input"

indices=($(seq 1 1 8)) 
run_experiments device_list[@] "$exp_config" "$dataset_name" "$epoch_mod1" "$epoch_mod2" "$seed" "${indices[@]}"