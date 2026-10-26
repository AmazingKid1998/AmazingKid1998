<details>
<summary><b>Pillar II: COMPASS — Refusal-Direction-Guided Mixed-Precision PTQ (First-Author Paper)</b></summary>
<br/>

Standard post-training quantisation (e.g. GPTQ) minimises layer-wise reconstruction error on a benign calibration set, and existing mixed-precision allocators decide which modules stay in FP16 using perplexity- or loss-based saliency. None of them anchor that choice to *where refusal is mechanistically encoded*. **COMPASS** (Cost-Matched Precision Allocation for Safe Systems) does: it ranks every linear module by how strongly it drives the model's refusal direction, then protects the most safety-critical modules under a strict parameter budget.

**1 · Refusal direction.** Following Arditi et al. (2024), the direction is the difference in mean final-token residual-stream activations between harmful prompts $`H`$ and benign prompts $`B`$ at layer $`\ell`$:

```math
r = \frac{1}{|H|}\sum_{x \in H} h_\ell(x) \;-\; \frac{1}{|B|}\sum_{x \in B} h_\ell(x), \qquad \hat{r} = \frac{r}{\lVert r \rVert}
```

The layer $`\ell`$ is chosen by causal evaluation (refusal drop when the direction is ablated on harmful probes, plus refusal rise when it is added on harmless probes), with extraction prompts kept disjoint from the probe set.

**2 · Multi-layer projection target.** A single-layer projection gives zero gradient to every layer after $`\ell`$ (about 31% of projection modules on Llama-3.1-8B), so the saliency target averages the projection over all post-extraction layers:

```math
\mathcal{L}(x) = \frac{1}{|K_\ell|}\sum_{k \in K_\ell} \left\langle h_k(x),\, \hat{r} \right\rangle, \qquad K_\ell = \lbrace \ell,\, \ell+1,\, \ldots,\, L-1 \rbrace
```

**3 · Per-module saliency (first-order Taylor proxy).** Because $`|\Delta \mathcal{L}| \approx |\nabla_W \mathcal{L} \cdot \Delta W|`$ and round-to-nearest error scales with $`|W|`$:

```math
s(W) = \mathbb{E}_{x \in H}\left[ \frac{1}{|W|} \sum_{i,j} \left| \frac{\partial \mathcal{L}(x)}{\partial W_{ij}} \right| \cdot \left| W_{ij} \right| \right]
```

**4 · Contrastive dual saliency.** Harmful and benign gradients are accumulated with opposing signs in one backward pass, so general-purpose modules (aligned gradients) cancel and context-dependent refusal modules survive:

```math
\nabla_W \mathcal{L}_{\text{dual}} = -\lambda_h \, \mathbb{E}_{x \in H}\left[\nabla_W \mathcal{L}(x)\right] + \lambda_b \, \mathbb{E}_{x \in B}\left[\nabla_W \mathcal{L}(x)\right]
```

```math
s_{\text{dual}}(W) = \frac{1}{|W|} \sum_{i,j} \left| \left(\nabla_W \mathcal{L}_{\text{dual}}\right)_{ij} \right| \cdot \left| W_{ij} \right|
```

**5 · Knapsack budget allocation.** With per-module parameter counts $`p(W)`$ and an FP16 parameter budget $`\mathcal{B}`$, the protected set is an exact 0/1 knapsack, solved by dynamic programming over 1M-parameter buckets (avoiding the small-module bias of greedy density selection):

```math
P^{\star} = \arg\max_{P \,:\, \sum_{W \in P} p(W) \,\le\, \mathcal{B}} \; \sum_{W \in P} s(W)
```

$`P^{\star}`$ is passed as `ignore=` to the GPTQ W4A16 modifier in `llm-compressor`, so those modules keep FP16 weights at per-Linear granularity.

**Key findings**

* **A distinct signal, not a perplexity proxy:** the top-5% refusal-saliency module set is nearly disjoint from perplexity saliency (Jaccard 0.04–0.14 across three models).
* **Saliency geometry of refusal:** saliency concentrates in attention `v_proj` / `o_proj`, and the safety-critical set transfers within an architecture family (Mistral ↔ Llama: Jaccard 0.50, Kendall-τ 0.71) but not across families (to Gemma: Jaccard 0.29).
* **Pre-registered cross-model robustness map:** ≥3-seed evaluation of Llama-3.1-8B, Mistral-7B-v0.3 and Gemma-7B-IT under W4A16 / W4A8 / NVFP4 shows robustness on English harmful-behaviour benchmarks, consistent with quantisation preserving the refusal direction (cosine ≈ 0.99); residual damage appears only cross-lingually (MultiJail-KO).
* **Toolchain-artifact correction:** an apparent Gemma × AWQ over-refusal and below-random MCQ collapse was traced to a known `llm-compressor` RMSNorm numerical bug (PR #2500), reproduced before the fix and shown to disappear after it.
* **Over-refusal as a first-class metric:** attack success rate, benign over-refusal and capability retention are evaluated jointly with seed-variance reporting.

</details>
