#!/bin/bash

# Directory containing the h5 files
dir="/home/gpadmin/tts/TensorFlowTTS/examples/fastspeech2_libritts/outdir_libri/checkpoints"

# Find the biggest number in the filenames
biggest_number=0
for file in "$dir"/model-*.h5; do
  filename=$(basename "$file")
  number="${filename#model-}"
  number="${number%.h5}"
  if [ "$number" -gt "$biggest_number" ]; then
    biggest_number="$number"
  fi
done

# Construct and execute the desired command
if [ "$biggest_number" -gt 0 ]; then
  checkpoint_path="$dir/ckpt-$biggest_number"
  model_path="$dir/model-$biggest_number.h5"

  CUDA_VISIBLE_DEVICES=0 python /home/gpadmin/tts/TensorFlowTTS/examples/fastspeech2_libritts/train_fastspeech2.py \
    --train-dir /home/gpadmin/tts/TensorFlowTTS/libritts/train/ \
    --dev-dir /home/gpadmin/tts/TensorFlowTTS/libritts/valid/ \
    --outdir /home/gpadmin/tts/TensorFlowTTS/examples/fastspeech2_libritts/outdir_libri/ \
    --config /home/gpadmin/tts/TensorFlowTTS/examples/fastspeech2_libritts/conf/fastspeech2libritts.yaml \
    --use-norm 1 \
    --f0-stat /home/gpadmin/tts/TensorFlowTTS/libritts/stats_f0.npy \
    --energy-stat /home/gpadmin/tts/TensorFlowTTS/libritts/stats_energy.npy \
    --mixed_precision 1 \
    --dataset_config /home/gpadmin/tts/TensorFlowTTS/preprocess/libritts_preprocess.yaml \
    --dataset_stats /home/gpadmin/tts/TensorFlowTTS/libritts/stats.npy \
    --dataset_mapping /home/gpadmin/tts/TensorFlowTTS/libritts/libritts_mapper.json \
    --resume "$checkpoint_path" \
    --pretrained "$model_path"
else
  echo "No h5 files found in $dir."
fi
