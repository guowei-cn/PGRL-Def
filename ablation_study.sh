#!/bin/bash

# Define the base parameters
POISON_TYPE="pattern"
CLASS=10
PB="poison"
PR=0.05
BASE_SAMPLE=10
BASE_AUG_N=6

# Run the first set of commands
echo "Running our_method_with_OT.py and our_method_without_OT.py with base parameters"
python our_method_with_OT.py -t "$POISON_TYPE" -class "$CLASS" -pb "$PB" -pr "$PR" -sample "$BASE_SAMPLE" -aug_n "$BASE_AUG_N"
python our_method_without_OT.py -t "$POISON_TYPE" -class "$CLASS" -pb "$PB" -pr "$PR" -sample "$BASE_SAMPLE" -aug_n "$BASE_AUG_N"

# Run for varying sample numbers
SAMPLE_NUMS=(10 6 3 1)
for sample_num in "${SAMPLE_NUMS[@]}"
do
    echo "Running our_method_with_OT.py with sample=$sample_num"
    python our_method_with_OT.py -t "$POISON_TYPE" -class "$CLASS" -pb "$PB" -pr "$PR" -sample "$sample_num" -aug_n "$BASE_AUG_N"
done

# Run for varying thresholds
THRESHOLDS=(0.9 0.6 0.3 0.0)
for th in "${THRESHOLDS[@]}"
do
    echo "Running our_method_with_OT.py with threshold=$th"
    python our_method_with_OT.py -t "$POISON_TYPE" -class "$CLASS" -pb "$PB" -pr "$PR" -sample "$BASE_SAMPLE" -aug_n "$BASE_AUG_N" -th "$th"
done

# Run for varying augmentation numbers
AUG_NUMS=(6 4 2 1)
for aug_n in "${AUG_NUMS[@]}"
do
    echo "Running our_method_with_OT.py with aug_n=$aug_n"
    python our_method_with_OT.py -t "$POISON_TYPE" -class "$CLASS" -pb "$PB" -pr "$PR" -sample "$BASE_SAMPLE" -aug_n "$aug_n"
done

echo "Execution completed for all configurations."
