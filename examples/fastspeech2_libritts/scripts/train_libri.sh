CUDA_VISIBLE_DEVICES=0 python examples/fastspeech2_libritts/train_fastspeech2.py \
  --train-dir ./libtitts/train/ \
  --dev-dir ./libtitts/valid/ \
  --outdir ./examples/fastspeech2_libritts/outdir_libri/ \
  --config ./examples/fastspeech2_libritts/conf/fastspeech2libritts.yaml \
  --use-norm 1 \
  --f0-stat ./libtitts/stats_f0.npy \
  --energy-stat ./libtitts/stats_energy.npy \
  --mixed_precision 1 \
  --dataset_config preprocess/libritts_preprocess.yaml \
  --dataset_stats ./libtitts/stats.npy


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
  --resume /home/gpadmin/tts/TensorFlowTTS/examples/fastspeech2_libritts/outdir_libri/checkpoints/ckpt-{the biggest number}
  --pretrained /home/gpadmin/tts/TensorFlowTTS/examples/fastspeech2_libritts/outdir_libri/checkpoints/model-{the biggest number}.h5


