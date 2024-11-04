#!/bin/bash

# Define arrays of values for each parameter
hole_counts=(1 3 5 7 10 12)
# hole_counts=(12)
hole_widths=(0.1 0.2 0.3 0.4)
# hole_widths=(0.4)

# Loop through each combination of parameters
for REPEAT in {1..3}; do
    for HOLE_COUNT in "${hole_counts[@]}"; do
        for HOLE_WIDTH in "${hole_widths[@]}"; do
            LOG_NAME="log_holecount_${HOLE_COUNT}_holewidth_${HOLE_WIDTH}_repeat_${REPEAT}"
            # Set environment variables
            export HOLE_COUNT
            export HOLE_WIDTH
            export LOG_NAME
            export REPEAT
            
            # Run the Python script with these environment variables
            python3 experimentor.py
        done
    done
done
