#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Array of configuration IDs
CONFIGS=(1 2 3 4 5)

echo "Starting sequential execution of 5 DRC Prediction configurations..."
echo "Logs will be recorded by Aim."

for ID in "${CONFIGS[@]}"; do
    echo "=========================================================="
    echo "Running Configuration ID: ${ID}"
    echo "=========================================================="

    echo "--> Step 1: Training Config ${ID}"
    python train.py --config-name="drc_train_${ID}"

    echo "--> Step 2: Testing Config ${ID}"
    # Wait to ensure the model checkpoint is written properly before testing
    sleep 2
    python test.py --config-name="drc_test_${ID}"

    echo "Configuration ${ID} completed successfully."
    echo ""
done

echo "All 5 configurations have been trained and tested!"
