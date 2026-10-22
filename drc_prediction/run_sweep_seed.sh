#!/bin/bash

# Seed sweep for the BiasedLoss + GroupNorm (BS_GN) configuration.
# Train + test only (no augmented metrics). Configs live in sweep_seed_config/.

set -e

SEEDS=(0 1 2)
CONFIG_DIR="sweep_seed_config"

for SEED in "${SEEDS[@]}"; do
    echo "=========================================================="
    echo "BS_GN seed ${SEED}"
    echo "=========================================================="
    echo "--> Training (seed ${SEED})"
    python train.py --config-path="${CONFIG_DIR}" --config-name="drc_train_BS_GN_seed${SEED}"

    echo "--> Testing (seed ${SEED})"
    python test.py --config-path="${CONFIG_DIR}" --config-name="drc_test_BS_GN_seed${SEED}"

    echo "Seed ${SEED} completed."
    echo ""
done

echo "Seed sweep finished."
