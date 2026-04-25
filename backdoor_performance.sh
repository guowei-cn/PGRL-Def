#!/bin/bash

# Define poison types
POISON_TYPES=("pattern" "ultrasonic" "adaptivecifar10" "freq_meg_500" "blto")

# Define poison ratios
POISON_RATIOS=(0.003 0.05)

# Loop through each poison type
for poison_type in "${POISON_TYPES[@]}"
do
    # Loop through each poison ratio for poisoned datasets
    for poison_ratio in "${POISON_RATIOS[@]}"
    do
        echo "Training with poison_type=$poison_type, poison_ratio=$poison_ratio"
        python train.py -t "$poison_type" -class 10 -pb poison -pr "$poison_ratio"
    done

    # Run training for benign datasets
    echo "Training with poison_type=$poison_type for benign dataset"
    python train.py -t "$poison_type" -class 10 -pb benign
done

echo "Training completed for all configurations."
