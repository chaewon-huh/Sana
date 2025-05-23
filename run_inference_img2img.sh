#!/bin/bash

# Sana Image-to-Image Inference Script

# Default settings
CONFIG="configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml"
CHECKPOINT="output/img2img_debug/checkpoints/latest.pth"
SOURCE_IMAGE=""
OUTPUT_DIR="output/inference_results"
STEPS=20
CFG_SCALE=1.0
SAMPLER="flow_euler"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --config CONFIG_PATH      Path to config file (default: $CONFIG)"
    echo "  --checkpoint CKPT_PATH    Path to checkpoint file (default: $CHECKPOINT)"
    echo "  --source SOURCE_PATH      Path to source image (required)"
    echo "  --output OUTPUT_DIR       Output directory (default: $OUTPUT_DIR)"
    echo "  --steps STEPS             Number of sampling steps (default: $STEPS)"
    echo "  --cfg_scale CFG_SCALE     CFG scale (default: $CFG_SCALE)"
    echo "  --sampler SAMPLER         Sampler type: flow_euler|dpm-solver (default: $SAMPLER)"
    echo "  --help                    Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --source data/FinalInputData256/example.png"
    echo "  $0 --source data/FinalInputData256/example.png --steps 28 --sampler dpm-solver"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --source)
            SOURCE_IMAGE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --cfg_scale)
            CFG_SCALE="$2"
            shift 2
            ;;
        --sampler)
            SAMPLER="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if source image is provided
if [[ -z "$SOURCE_IMAGE" ]]; then
    echo "Error: Source image is required!"
    echo ""
    show_usage
    exit 1
fi

# Check if source image exists
if [[ ! -f "$SOURCE_IMAGE" ]]; then
    echo "Error: Source image '$SOURCE_IMAGE' does not exist!"
    exit 1
fi

# Check if checkpoint exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint '$CHECKPOINT' does not exist!"
    exit 1
fi

# Create output filename based on source image name
SOURCE_BASENAME=$(basename "$SOURCE_IMAGE" | cut -d. -f1)
OUTPUT_PATH="$OUTPUT_DIR/${SOURCE_BASENAME}_generated.png"

echo "=== Sana Image-to-Image Inference ==="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Source: $SOURCE_IMAGE"
echo "Output: $OUTPUT_PATH"
echo "Steps: $STEPS"
echo "CFG Scale: $CFG_SCALE"
echo "Sampler: $SAMPLER"
echo "=================================="

# Run inference
python inference_img2img.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --source "$SOURCE_IMAGE" \
    --output "$OUTPUT_PATH" \
    --steps "$STEPS" \
    --cfg_scale "$CFG_SCALE" \
    --sampler "$SAMPLER"

echo "Inference complete! Output saved to: $OUTPUT_PATH" 