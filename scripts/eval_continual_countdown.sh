#!/bin/bash

# Default paths - These should match Docker volume mounts
METRICS_DIR=${METRICS_DIR:-"/app/metrics"}
PLOTS_DIR=${PLOTS_DIR:-"/app/plots"}
LOGS_DIR=${LOGS_DIR:-"/app/logs"}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --metrics-dir)
            METRICS_DIR="$2"
            shift 2
            ;;
        --plots-dir)
            PLOTS_DIR="$2"
            shift 2
            ;;
        --logs-dir)
            LOGS_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Ensure directories exist and have correct permissions
mkdir -p "$METRICS_DIR" "$PLOTS_DIR" "$LOGS_DIR"
chmod -R 777 "$METRICS_DIR" "$PLOTS_DIR" "$LOGS_DIR"

echo "Starting evaluation with configuration:"
echo "METRICS_DIR: $METRICS_DIR"
echo "PLOTS_DIR: $PLOTS_DIR"
echo "LOGS_DIR: $LOGS_DIR"

# Check if training logs exist
if [ ! -d "$LOGS_DIR" ] || [ -z "$(ls -A $LOGS_DIR)" ]; then
    echo "Error: No training logs found in $LOGS_DIR"
    echo "Please ensure training has completed and generated log files."
    exit 1
fi

# Check if model checkpoints exist
checkpoint_count=$(find "$METRICS_DIR" -type d -name "initial_*" -o -name "final_*" | wc -l)
if [ "$checkpoint_count" -eq 0 ]; then
    echo "Error: No model checkpoints found in $METRICS_DIR"
    echo "Please ensure training has completed and generated model checkpoints."
    exit 1
fi

# Run evaluation
python3 experiments/continual/eval.py \
    --metrics-dir "$METRICS_DIR" \
    --plots-dir "$PLOTS_DIR" \
    --logs-dir "$LOGS_DIR"

# Check if evaluation was successful
if [ $? -eq 0 ]; then
    # Fix permissions for generated files
    chmod -R 777 "$PLOTS_DIR" "$METRICS_DIR"
    
    # Check if files were generated
    if [ -f "$PLOTS_DIR/training_metrics.png" ] && [ -f "$METRICS_DIR/consolidated_metrics.json" ]; then
        echo "Evaluation completed successfully!"
        echo "Results can be found in:"
        echo "  - Plots: $PLOTS_DIR/training_metrics.png"
        echo "  - Metrics: $METRICS_DIR/consolidated_metrics.json"
    else
        echo "Warning: Evaluation completed but some output files are missing."
        echo "Please check the evaluation logs for details."
    fi
else
    echo "Error: Evaluation failed!"
    exit 1
fi
