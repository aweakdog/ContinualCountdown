#!/bin/bash

# Activate conda environment
. /opt/conda/etc/profile.d/conda.sh
conda activate zero

# Default paths - These should match Docker volume mounts
METRICS_DIR=${METRICS_DIR:-"/app/metrics"}
PLOTS_DIR=${PLOTS_DIR:-"/app/plots"}
LOGS_DIR=${LOGS_DIR:-"/app/logs"}

# Default to evaluating all models
EVAL_0_5B=1
EVAL_1_5B=1
EVAL_3B=1

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
        --model-size)
            case "$2" in
                0.5b)
                    EVAL_0_5B=1
                    EVAL_1_5B=0
                    EVAL_3B=0
                    ;;
                1.5b)
                    EVAL_0_5B=0
                    EVAL_1_5B=1
                    EVAL_3B=0
                    ;;
                3b)
                    EVAL_0_5B=0
                    EVAL_1_5B=0
                    EVAL_3B=1
                    ;;
                *)
                    echo "Invalid model size: $2. Must be '0.5b', '1.5b', or '3b'"
                    exit 1
                    ;;
            esac
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
#checkpoint_count=$(find "$METRICS_DIR" -type d -name "initial_*" -o -name "final_*" | wc -l)
#if [ "$checkpoint_count" -eq 0 ]; then
#    echo "Error: No model checkpoints found in $METRICS_DIR"
#    echo "Please ensure training has completed and generated model checkpoints."
#    exit 1
#fi

# Run evaluation for each model size
if [ "$EVAL_0_5B" -eq 1 ]; then
    echo "\nEvaluating Qwen 0.5B model..."
    python3 experiments/continual/eval.py \
        --model-size 0.5b \
        --metrics-dir "$METRICS_DIR" \
        --plots-dir "$PLOTS_DIR" \
        --logs-dir "$LOGS_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Error evaluating 0.5B model"
        exit 1
    fi
    
    # Fix permissions for generated files
    chmod -R 777 "$PLOTS_DIR" "$METRICS_DIR"
    
    # Check if files were generated
    if [ -f "$PLOTS_DIR/training_metrics_0.5b.png" ] && [ -f "$METRICS_DIR/consolidated_metrics_0.5b.json" ]; then
        echo "Qwen 0.5B evaluation completed successfully!"
        echo "Results can be found in:"
        echo "  - Plots: $PLOTS_DIR/training_metrics_0.5b.png"
        echo "  - Metrics: $METRICS_DIR/consolidated_metrics_0.5b.json"
    else
        echo "Warning: Qwen 0.5B evaluation completed but some output files are missing."
        echo "Please check the evaluation logs for details."
    fi
fi

if [ "$EVAL_1_5B" -eq 1 ]; then
    echo "\nEvaluating Qwen 1.5B model..."
    python3 experiments/continual/eval.py \
        --model-size 1.5b \
        --metrics-dir "$METRICS_DIR" \
        --plots-dir "$PLOTS_DIR" \
        --logs-dir "$LOGS_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Error evaluating 1.5B model"
        exit 1
    fi
    
    # Fix permissions for generated files
    chmod -R 777 "$PLOTS_DIR" "$METRICS_DIR"
    
    # Check if files were generated
    if [ -f "$PLOTS_DIR/training_metrics_1.5b.png" ] && [ -f "$METRICS_DIR/consolidated_metrics_1.5b.json" ]; then
        echo "Qwen 1.5B evaluation completed successfully!"
        echo "Results can be found in:"
        echo "  - Plots: $PLOTS_DIR/training_metrics_1.5b.png"
        echo "  - Metrics: $METRICS_DIR/consolidated_metrics_1.5b.json"
    else
        echo "Warning: Qwen 1.5B evaluation completed but some output files are missing."
        echo "Please check the evaluation logs for details."
    fi
fi

if [ "$EVAL_3B" -eq 1 ]; then
    echo "\nEvaluating Qwen 3B model..."
    python3 experiments/continual/eval.py \
        --model-size 3b \
        --metrics-dir "$METRICS_DIR" \
        --plots-dir "$PLOTS_DIR" \
        --logs-dir "$LOGS_DIR"
    
    if [ $? -ne 0 ]; then
        echo "Error evaluating 3B model"
        exit 1
    fi
    
    # Fix permissions for generated files
    chmod -R 777 "$PLOTS_DIR" "$METRICS_DIR"
    
    # Check if files were generated
    if [ -f "$PLOTS_DIR/training_metrics_3b.png" ] && [ -f "$METRICS_DIR/consolidated_metrics_3b.json" ]; then
        echo "Qwen 3B evaluation completed successfully!"
        echo "Results can be found in:"
        echo "  - Plots: $PLOTS_DIR/training_metrics_3b.png"
        echo "  - Metrics: $METRICS_DIR/consolidated_metrics_3b.json"
    else
        echo "Warning: Qwen 3B evaluation completed but some output files are missing."
        echo "Please check the evaluation logs for details."
    fi
fi
