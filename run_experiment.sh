#!/bin/bash

# Define arrays of values for each parameter
hole_counts=(7)
hole_widths=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)

# Loop through each combination of parameters
for HOLE_COUNT in "${hole_counts[@]}"; do
    for HOLE_WIDTH in "${hole_widths[@]}"; do
        LOG_NAME="log_holecount_${HOLE_COUNT}_holewidth_${HOLE_WIDTH}"
        # Set environment variables
        export HOLE_COUNT
        export HOLE_WIDTH
        export LOG_NAME
        
        # Run the Python script with these environment variables
        python3 experimentor.py
    done
done
