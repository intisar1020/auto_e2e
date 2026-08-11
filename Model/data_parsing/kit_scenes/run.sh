#!/usr/bin/env bash
# ==============================================================================
# Helper script to launch training and generate trajectory prediction videos
# Usage:
#   cd Model/data_parsing/kit_scenes
#   bash run.sh train      # Launch training (train_main.py)
#   bash run.sh video      # Generate trajectory prediction video from extracted scene dir
#   bash run.sh video-tar  # Generate trajectory prediction video from scene .tar archive
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"

# Configure PyTorch memory allocation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

case "$MODE" in
  train)
    echo "================================================================="
    echo " Launching Training (50 epochs, 16 channels live)..."
    echo "================================================================="
    EXP_NAME="${2:-exp-3-route-consistency}"
    python train_main.py \
      --epochs 50 \
      --grad-accum 4 \
      --lr 1e-4 \
      --max-grad-norm 5.0 \
      --exp-name "$EXP_NAME"
    ;;

  video)
    echo "================================================================="
    echo " Generating Trajectory Prediction Video from Extracted Scene Dir..."
    echo "================================================================="
    SCENE_DIR="${2:-datasets/train/00efe646-1e8f-cadc-d642-e676781187ef}"
    OUTPUT_VIDEO="${3:-trajectory_video.mp4}"
    CKPT="${4:-exp-2-baseline/checkpoints/best.pt}"

    python generate_trajectory_video_dir.py \
      --checkpoint "$CKPT" \
      --scene-dir "$SCENE_DIR" \
      --output "$OUTPUT_VIDEO" \
      --fps 3 \
      --horizon 6.4
    ;;

  video-tar)
    echo "================================================================="
    echo " Generating Trajectory Prediction Video from Scene .tar Archive..."
    echo "================================================================="
    TAR_PATH="${2:-data/73197a6d-fd55-2fd2-4a47-ddb3ff3b7db7.tar}"
    OUTPUT_VIDEO="${3:-trajectory_video_tar.mp4}"
    CKPT="${4:-exp-2-baseline/checkpoints/best.pt}"

    python generate_trajectory_video.py \
      --checkpoint "$CKPT" \
      --tar "$TAR_PATH" \
      --output "$OUTPUT_VIDEO" \
      --fps 3 \
      --horizon 6.4
    ;;

  *)
    echo "Usage: bash run.sh [train | video | video-tar]"
    echo ""
    echo "Commands:"
    echo "  bash run.sh train [exp_name]"
    echo "      Train the model (50 epochs, all 16 channels)."
    echo "      Default exp_name : exp-3-route-consistency"
    echo ""
    echo "  bash run.sh video [scene_dir] [output_mp4] [checkpoint]"
    echo "      Generate video from an extracted scene directory."
    echo "      Default scene_dir : datasets/train/00efe646-1e8f-cadc-d642-e676781187ef"
    echo "      Default output    : trajectory_video.mp4"
    echo "      Default checkpoint: exp-2-baseline/checkpoints/best.pt"
    echo ""
    echo "  bash run.sh video-tar [tar_path] [output_mp4] [checkpoint]"
    echo "      Generate video from a .tar archive file."
    echo "      Default tar_path  : data/73197a6d-fd55-2fd2-4a47-ddb3ff3b7db7.tar"
    echo "      Default output    : trajectory_video_tar.mp4"
    echo "      Default checkpoint: exp-2-baseline/checkpoints/best.pt"
    exit 1
    ;;
esac
