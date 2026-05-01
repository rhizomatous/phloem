# TODO

## in progress

- [ ] 1B-token 32× all-layers training run on Gemma 4 E2B (running on RunPod)

## next up

- [ ] run eval.py on the completed 1B checkpoint
- [ ] run max_activations.py on the completed 1B checkpoint (10M token scan)
- [ ] browse features with browse_features.py, eyeball quality
- [ ] upload SAE checkpoints + max activation parquet files to HuggingFace

## research questions

- [ ] count dead features in EleutherAI's sae-llama-3.1-8b-32x per layer.
  load their published checkpoints, run 10M tokens of RedPajama, compare
  per-layer dead % against our Gemma 4 E2B results. if the mid-network spike
  disappears on Llama (uniform architecture), that's evidence it's driven by
  Gemma 4's MLP width transition. same library, same dataset, same expansion
  factor — clean comparison.
- [ ] investigate Active Subspace Initialization (arXiv:2508.16929) for
  layers 16-18. run PCA on those layers' activations, initialize SAE into the
  principal subspace, see if dead % drops. requires modifying sparsify.
- [ ] try per-layer expansion factor: 16× at layers 15-21, 32× elsewhere.
  match SAE capacity to effective dimensionality.
- [ ] exclude BOS token (token ID 2) via TrainConfig.exclude_tokens. BOS
  activations have unusually high norms and can distort feature learning.
  free intervention, no tradeoff.
- [ ] try lower learning rate. signum auto-computes 2.89e-3 for 32×. anecdotal
  evidence from Qwen BatchTopK training: lowering LR from 3e-4 to 5e-5
  eliminated dead features entirely. signum and adam aren't directly comparable
  but worth experimenting with --lr and/or switching to adam.
- [ ] try k warmup via k_decay_steps. start at higher k (less competition),
  decay to target k=100. gives weaker features time to establish before
  competition tightens.

## future stages

- [ ] Gemma 4 E4B: d_in=2560, 42 layers, ~20 GB model. 8× all-layers fits
  A100 80GB. 32× needs layer subset or 2× A100.
- [ ] Gemma 4 31B: d_in=5376, 60 layers, ~61 GB model. needs A100 80GB,
  tight on VRAM.
- [ ] Kimi K2.5: MoE, d_in=7168, 61 layers, ~1T params. needs 8× A100.

## tooling

- [ ] autointerp: local model for bulk explanation + Claude API for hard cases
- [ ] static feature browser (Neuronpedia-style) or HF dataset viewer
- [ ] streaming-to-disk tokenization for >10B token runs (write directly to
  memmap without holding full dataset in RAM)
- [ ] wire up micro_acc_steps as CLI arg (reduces SAE peak memory, could allow
  larger batch_size)
