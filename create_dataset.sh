#!/bin/bash

# Define the base directory for the poisonDataset folder
BASE_DIR="poisonDataset"

# Define the subfolders
SUBFOLDERS=("pattern" "ultrasonic" "adaptivecifar10" "freq_meg_500" "blto")

# Loop over each subfolder
for folder in "${SUBFOLDERS[@]}"
do
    # Navigate to the subfolder
    cd "$BASE_DIR/$folder" || { echo "Directory $BASE_DIR/$folder not found!"; exit 1; }

    # Run the command for both poison ratios
    echo "Running for poison_ratio 0.003 in $folder"
    python createDataset.py --poison_ratio 0.003

    echo "Running for poison_ratio 0.05 in $folder"
    python createDataset.py --poison_ratio 0.05

    # Go back to the base directory
    cd - || exit
done

echo "Dataset generation completed for all subfolders."
