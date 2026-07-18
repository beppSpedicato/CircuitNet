#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Array of configuration IDs
CONFIGS=('baseline' 25 50 75)

echo "Starting sequential execution of 3 DRC Prediction configurations..."
echo "Logs will be recorded by Aim."

for ID in "${CONFIGS[@]}"; do
    echo "=========================================================="
    echo "Running Configuration ID: ${ID}"
    echo "=========================================================="
    echo "--> Training Config ${ID}"
    python train.py --config-name="drc_train_th_${ID}"

    sleep 2  # Add a short pause between training and testing
    
    echo "--> Testing Config ${ID}"
    python test.py --config-name="drc_test_th_${ID}"

    echo "Configuration ${ID} completed successfully."
    echo ""
done

echo "All 5 configurations have been trained and tested!"
