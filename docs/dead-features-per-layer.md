# per-layer dead feature variation in SAE training

## what we observed

training 32× TopK SAEs (k=100, multi_topk=True) on all 35 layers of Gemma 4 E2B with 1B tokens from RedPajama v2, we see a characteristic per-layer pattern in dead feature percentages despite uniform FVU (~0.15) across all layers:

- layers 0-14: healthy, low dead %
- layers 15-19: trending upward
- layers 16-18: plateaued at 60-70% dead
- layers 20-21: plateaued at ~30% dead
- layers 22-34: very low dead %, doing great

the SAE reconstructs equally well everywhere — layers 16-18 just do it with 30-40% of features while other layers use 80-90%+.

## Gemma 4 E2B architectural context

- MLP intermediate size doubles from 6,144 to 12,288 at layer 15
- global attention every 5 layers (4, 9, 14, 19, 24, 29, 34), rest are sliding-window (512 tokens)
- d_in = 1,536 constant across all layers
- Per-Layer Embeddings (PLE) at every layer

## why this happens: mid-network dimensionality dip

### intrinsic dimensionality compression at mid-depth

Valeriani et al. (2023) measured intrinsic dimensionality (ID) across transformer layers and found a consistent pattern across models: expansion in early layers, compression in middle layers, then task-specific shaping in later layers. our dead-feature spike at layers 15-21 lands exactly where this mid-network compression is expected.

- [Valeriani et al., "The geometry of hidden representations of large transformer models" (2023)](https://arxiv.org/abs/2302.00294)

### dimensional collapse in attention outputs

a February 2026 paper demonstrates that attention outputs are confined to ~60% effective dimensionality while MLP outputs and residual streams sit at ~90%. randomly initialized SAE features that fall outside the active subspace go dead because there's no gradient signal to pull them in.

their fix — **Active Subspace Initialization** — projects SAE encoder/decoder weights into the principal subspace of each layer's activations before training. this reduced dead features from 87% to below 1%.

- [Dimensional Collapse in Transformer Attention Outputs (2026)](https://arxiv.org/abs/2508.16929)

### Gemma Scope layer-dependent behavior

Anthropic's Gemma Scope trained JumpReLU SAEs on every layer of Gemma 2 2B/9B but sampled detailed hyperparameter sweeps at only 4 depth fractions (25%, 50%, 65%, 85%), implying they observed layer-dependent behavior warranting different tuning per depth. they don't publish explicit per-layer dead-feature curves.

- [Gemma Scope (Lieberum et al., 2024)](https://arxiv.org/abs/2408.05147)

### Gao et al. layer selection

OpenAI reported that without mitigations, up to 90% of latents can die, and strategically chose layers ~5/6 of the way through GPT-4 for their main experiments, implicitly acknowledging layer-dependent difficulty.

- [Gao et al., "Scaling and evaluating sparse autoencoders" (2024)](https://arxiv.org/abs/2406.04093)

## what amplifies it in Gemma 4 E2B

the mid-network dimensionality dip is likely amplified by the MLP width doubling at layer 15. the residual stream geometry changes at that boundary, and the layers immediately after (16-18) are where the adjustment is sharpest. by layer 22, the representations have stabilized in the new regime.

no published work exists on SAE quality variation in hybrid-attention models (sliding-window + global). our per-layer dead-feature pattern appears to be novel in this regard.

## possible interventions for future runs

- **Active Subspace Initialization**: run PCA on each layer's activations and initialize the SAE encoder/decoder into that subspace before training. most promising intervention per the literature, but requires modifying sparsify.
- **per-layer expansion factor**: use 16× at layers 15-21 and 32× elsewhere, matching SAE capacity to the actual dimensionality of each layer's activations.
- **per-layer k**: lower k at mid-network layers where fewer features are needed.
- **accept it**: uniform FVU means the SAE is working correctly. the dead features are telling you about the model's architecture, not about a training failure. this is itself an interesting finding.

## why Gemma 4 shows this more than Gemma 2

Gemma Scope trained SAEs on every layer of Gemma 2 without reporting dramatic per-layer dead-feature variation. Gemma 4's architecture has several features Gemma 2 lacked that likely drive the pattern we see:

| | Gemma 2 | Gemma 4 |
|---|---|---|
| **Modality** | text-only | trimodal (text + vision + audio) |
| **Attention** | uniform head_dim across all layers | hybrid: head_dim=256 (sliding-window) vs 512 (global) |
| **KV heads** | uniform across layers | varies: more KV heads for sliding-window, fewer for global |
| **MLP width** | uniform across layers | doubles mid-network (6,144 → 12,288 in E2B) |
| **PLE** | no | yes — Per-Layer Embeddings add ~45% of total params |
| **Global attention** | alternating (every other layer) | sparse (every 5th-6th layer) |
| **K=V trick** | no | yes, in global layers (keys = values, halves KV cache) |
| **RoPE** | standard | dual RoPE (different theta for sliding vs global) + partial rotary (25% of dims in global layers) |

- **MLP width is non-uniform.** Gemma 2 used the same MLP width at every layer. Gemma 4 E2B doubles from 6,144 to 12,288 at layer 15. this is the most likely driver of the mid-network dead-feature spike — the residual stream geometry changes at that boundary.
- **global attention is much sparser.** Gemma 2 alternated global/local every other layer. Gemma 4 uses global attention only every 5th layer, creating long stretches of sliding-window-only layers with restricted context.
- **hybrid head dimensions.** Gemma 2 used uniform head_dim across all layers. Gemma 4 uses head_dim=256 for sliding-window and head_dim=512 for global layers, with different KV head counts. different layer types contribute differently to the residual stream.
- **Per-Layer Embeddings (PLE).** entirely new in Gemma 4. each layer receives a unique conditioning vector from a shared embedding table, meaning each layer's residual stream has a different bias/offset. this could contribute to per-layer variation in activation distributions.
- **dual RoPE.** sliding-window layers use theta=10000, global layers use theta=1000000 with partial_rotary_factor=0.25. different positional encoding regimes at different layers.

Gemma 2's architectural uniformity across layers meant SAEs could use the same hyperparameters everywhere without issues. Gemma 4's heterogeneous layer structure means per-layer SAE behavior varies, and the mid-network MLP transition is where it shows most.

## further reading

- [Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021)](https://transformer-circuits.pub/2021/framework/index.html) — residual stream framework
- [Anthropic, Crosscoders (2024)](https://transformer-circuits.pub/2024/crosscoders/index.html)
- [Gemma Scope 2 Technical Paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf)
