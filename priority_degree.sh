#!/bin/bash

# Define the poisoning types as an array "pattern"
poison_types=("pattern" "ultrasonic" "adaptivecifar10" "freq_meg_500")

# Define the pr values as an array
pr_values=(0.003 0.05)

# Loop through each poisoning type and each pr value
for poison_type in "${poison_types[@]}"; do
    for pr in "${pr_values[@]}"; do
        echo "Running with poison_type: $poison_type and pr: $pr"
        python visualize_boxplot_first10epoch.py \
            -t "$poison_type" \
            -class 10 \
            -pb poison \
            -d cuda:0 \
            -pr "$pr" \
            #-trap True
    done
done
