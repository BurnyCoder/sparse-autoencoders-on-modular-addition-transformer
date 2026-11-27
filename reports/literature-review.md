# Literature review

Author: Libor "Burny" Burian. Date: 2026-11-27. Searches run on 2026-11-27 (web, arXiv, Hugging Face Hub).

Scope: what is known about (1) the grokked modular-addition transformer and its algorithm, (2) sparse autoencoders (SAEs), (3) how SAEs represent circular / multi-dimensional features, and (4) steering and intervention methods and their baselines. Each entry lists what this project reuses. Facts were extracted from the papers' HTML full text by reading agents; items that came only from a summary of a truncated page, not verbatim text, are marked *(unverified detail)* and are not relied on for design decisions.

## 1. Grokking and the modular-addition algorithm

**Power et al. 2022, "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"** ([arXiv 2201.02177](https://arxiv.org/abs/2201.02177)). Defines grokking: validation accuracy "suddenly begins to increase from chance level toward perfect generalization" long after training accuracy saturates. On modular division mod 97 with 50% training data, validation accuracy needed about 1000× more optimization steps than training accuracy. Weight decay is reported as particularly effective at reducing the data needed.
*Reused:* the task family and the AdamW + weight-decay recipe (lr 1e-3, wd 1, β2 = 0.98).

**Nanda, Chan, Lieberum, Smith, Steinhardt 2023, "Progress measures for grokking via mechanistic interpretability"** (ICLR 2023, [arXiv 2301.05217](https://arxiv.org/abs/2301.05217); code/figures: <https://neelnanda.io/grokking-paper>). The model this project studies:
- 1-layer ReLU transformer, d_model 128, 4 heads of dimension 32, 512 MLP neurons, no LayerNorm, untied embed/unembed, learned positional embeddings. Input "a b =" with p = 113, trained on 30% of all pairs with full-batch AdamW (lr 0.001, weight decay 1) for 40,000 epochs. Grokking happens after about 10,000 epochs.
- Key frequencies {14, 35, 41, 42, 52} in the mainline model, read off the Fourier-transformed embedding matrix ("only has significant nonnegligible norm at 6 frequencies"; 5 are used).
- MLP neurons: "out of 512 total neurons, 433 (84.6%) have over 85% of their variance explained with a single frequency", each approximated by a degree-2 polynomial of sin/cos of a and b at one frequency. Neurons cluster by frequency, and for each cluster the neuron-to-logit map W_L = W_U W_out has only sin/cos components at that frequency.
- "cos(w_k(a+b)), sin(w_k(a+b)) are linearly represented in the MLP activations". Logits are approximated by Σ_k α_k cos(w_k(a+b−c)), which "explains 95% of the variance".
- Interventions: *restricted loss* keeps only the key-frequency components of the logits (via a 2D DFT over (a, b)), and *excluded loss* removes them. Ablating all non-key Fourier components *improves* loss (to 7.24·10⁻⁸). Projecting MLP activations onto the 10 W_L directions decreases loss by 50%, while projecting onto their null space gives loss 5.27 ("worse than uniform").
- Training phases: memorization, circuit formation, cleanup.
- Attention: "two of the attention heads approximately compute degree-2 polynomials of sines and cosines of a particular frequency (and the other two are used to increase the magnitude of the input embeddings in the residual stream)".

*Reused:* the architecture, key-frequency rule, neuron-FVE metric, restricted/excluded loss, and the logit formula as ground truth for scoring SAE features.

**TransformerLens Grokking demo** ([notebook](https://github.com/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb)). The official TransformerLens reproduction. In TransformerLens 4.x it builds the model with `TransformerBridge.boot_native`: p = 113, '=' token id 113, 30% train split from `torch.randperm` under DATA_SEED = 598, model seed 999, biases frozen at 0 (`requires_grad = False` for every `*bias`), AdamW lr 1e-3, wd 1.0, betas (0.9, 0.98), 25,000 full-batch epochs, checkpoints every 100 epochs, float64 log-softmax loss at the last position.
*Reused:* this exact recipe, extended to the paper's 40,000 epochs.

**User's prior checkpoint** ([`BurnyCoder/grokking-modular-addition-transformer`](https://huggingface.co/BurnyCoder/grokking-modular-addition-transformer)). This is a TransformerLens 3 `HookedTransformer` trained with the demo recipe (25k epochs). Its model card reports key frequencies 9, 33, 36, 38, 55 and 298/512 neurons with > 85% variance at one frequency. It is a secondary (non-peer-reviewed) source, used only as a cross-check.

**Zhong, Liu, Tegmark, Andreas 2023, "The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks"** ([arXiv 2306.17844](https://arxiv.org/abs/2306.17844); code <https://github.com/fjzzq2002/pizza>). Two algorithms for modular addition, with E_a ≈ (cos w_k a, sin w_k a):
- **Clock** multiplies the inputs (angle addition), giving logits cos(w_k(a+b−c)).
- **Pizza** averages the embeddings, E_ab = (E_a + E_b)/2, then uses the ReLU MLP to map E_ab to |cos(w_k(a−b)/2)|·(cos w_k(a+b), sin w_k(a+b)), giving logits |cos(w_k(a−b)/2)|·cos(w_k(a+b−c)). "Pizza requires only absolute value computation, which is easily implemented by the ReLU layers."
- Diagnostics: *gradient symmetricity* (cosine similarity of ∂Q/∂E_a and ∂Q/∂E_b over 100 random inputs; symmetric means Pizza) and *distance irrelevance* (whether logits at the correct answer depend on a−b; Pizza depends on it).
- Their constant-attention transformer (Model A, equivalent to a ReLU MLP) learns Pizza. The standard 1-layer ReLU transformer (Model B) learns Clock, and they state that "Nanda et al. find that neural networks like our Model B implement this Clock algorithm". Interpolating attention strength shifts the phase between the two algorithms *(unverified detail: exact thresholds)*.

*Reused:* both algorithms as competing hypotheses whose steering predictions differ; the symmetricity and distance diagnostics; and the constant-attention control model. Because Zhong et al. report that standard-attention transformers like Nanda's learn Clock, the Pizza-derived predictions in our preregistration (antipodal invariance, half-angle transport) have an uncertain prior. They are discriminating tests, not expectations.

**Chughtai, Chan, Nanda 2023, "A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations"** ([arXiv 2302.03025](https://arxiv.org/abs/2302.03025); code <https://github.com/bilal-chughtai/rep-theory-mech-interp>). Generalizes Nanda's Fourier algorithm to group composition via representation theory ("group composition via representations"); Nanda's algorithm is the C_113 special case. They report evidence "for weak universality, but against strong universality": which representations are learned varies across seeds.
*Reused:* motivation for treating each base-model seed (with its own key frequencies) as the unit of inference.

**Gromov 2023, "Grokking modular arithmetic"** ([arXiv 2301.02679](https://arxiv.org/abs/2301.02679)). An analytic solution for a 2-layer MLP with quadratic activation: first-layer weights cos(2πkn/p + φ) and readout weights with phase constraint φ1 + φ2 = φ3, so constructive interference makes a delta at n + m ≡ q (mod p). "No explicit regularization is necessary for grokking to occur."
*Reused:* the constructive-interference picture behind multi-frequency steering.

**Lin 2025, "Feature Identification via the Empirical NTK"** ([arXiv 2510.00468](https://arxiv.org/abs/2510.00468)). On a smaller 1-layer transformer (p = 29, d_model 64, 256 MLP units), the top eNTK eigendirections at the attention-OV and MLP-input layers overlap the "A+B" family {cos(w_k a)+cos(w_k b), sin(w_k a)+sin(w_k b)} (score 3.86–3.90 of 4). The MLP-output and unembedding layers overlap the "sum" family cos/sin(w_k(a+b)) (3.94–3.97 of 4). The paper's results are correlational, and it notes that "a 65k-token dataset is likely insufficient to produce a training signal for a SAE".
*Reused:* the A+B hypothesis for the pre-MLP state, which implies the cone geometry z_k = e^{iθa} + e^{iθb} = 2cos(d/2)·e^{is/2}. That geometry underlies preregistered hypotheses H1–H5.

**Further grokking-mechanism work, context only:**
- **Manning-Coe et al. 2025, "Grokking vs. Learning: Same Features, Different Encodings"** ([arXiv 2502.01739](https://arxiv.org/abs/2502.01739)): grokked and steadily trained models learn the same features with different compression.
- **He, Wang, Chen, Yang 2026, "On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking"** ([arXiv 2602.16849](https://arxiv.org/abs/2602.16849); code <https://github.com/Y-Agent/modular-addition-feature-learning>): 2-layer MLPs with phase symmetry and majority voting, and a lottery-ticket competition between frequencies inside each neuron.

## 2. Sparse autoencoders

**Bricken et al. 2023, "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"** ([Transformer Circuits](https://transformer-circuits.pub/2023/monosemantic-features)). SAEs (L2 reconstruction + L1 penalty on the hidden activations) trained on the 512-neuron MLP activations of a *one-layer* transformer from 8 billion data points, with dictionaries from 512 to 131,072 features. The page exceeded our fetcher's size limit, so these facts come from the [Anthropic summary](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning) and secondary sources.
*Reused:* the precedent of SAEs on 1-layer MLP activations.

**Cunningham et al. 2023, "Sparse Autoencoders Find Highly Interpretable Features in Language Models"** ([arXiv 2309.08600](https://arxiv.org/abs/2309.08600); code <https://github.com/HoagyC/sparse_coding>). c = ReLU(Mx + b), x̂ = Mᵀc, loss ‖x − x̂‖² + α‖c‖₁, with row-normalized M. Introduces evaluation by splicing the reconstruction into the model, plus auto-interpretability.

**Gao et al. 2024, "Scaling and evaluating sparse autoencoders"** ([arXiv 2406.04093](https://arxiv.org/abs/2406.04093); code <https://github.com/openai/sparse_autoencoder>).
- TopK SAE: z = TopK(W_enc(x − b_pre)), x̂ = W_dec z + b_pre, trained on MSE only, which "removes the need for the L1 penalty".
- Dead latents are handled by initializing the encoder as the decoder transpose and by an auxiliary loss (AuxK) on dead latents.
- Metrics: normalized MSE, L0, downstream ΔCE/KL, probe loss, and explanation precision/recall.

**Bussmann, Leask, Nanda 2024, "BatchTopK Sparse Autoencoders"** ([arXiv 2412.06410](https://arxiv.org/abs/2412.06410); code <https://github.com/bartbussmann/BatchTopK>).
- Selects "the top n×k activations across the entire batch of n samples", so L0 varies per input but averages k.
- At inference, a JumpReLU threshold θ = E[min positive activation] replaces the batch dependence.
- Hyperparameters: Adam lr 3·10⁻⁴, β = (0.9, 0.99), batch 4096, k ∈ {16, 32, 64}, AuxK coefficient 1/32 with k_aux = 512.
- Reported performance is comparable to or better than TopK and JumpReLU on NMSE and CE degradation, though mixed in places.

*Reused:* the primary SAE architecture.

**Rajamanoharan et al. 2024, "Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders"** ([arXiv 2407.14435](https://arxiv.org/abs/2407.14435)). JumpReLU_θ(z) = z·H(z − θ) with per-feature thresholds, an L0 penalty trained via straight-through estimators, and FVU as a secondary metric.
*Reused:* ablation architecture.

**Braun et al. 2024, "Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning"** ([arXiv 2405.12241](https://arxiv.org/abs/2405.12241); code <https://github.com/ApolloResearch/e2e_sae>). End-to-end SAEs trained on the KL between the model's outputs with and without the SAE spliced in, plus a sparsity penalty, with the model frozen. They reach similar CE loss with fewer active features.
*Reused:* optional ablation, motivated by our prediction that most mlp_post variance is logit-null.

**Karvonen et al. 2025, "SAEBench"** ([arXiv 2503.09532](https://arxiv.org/abs/2503.09532); code <https://github.com/adamkarvonen/SAEBench>). Eight metrics covering concept detection, interpretability, reconstruction (loss recovered (H* − H0)/(H_orig − H0)) and disentanglement. Key finding: "gains on proxy metrics do not reliably translate to better practical performance".
*Reused:* the loss-recovered definition, and the motivation to evaluate SAEs on downstream behavior (logit FVU, KL, steering), not reconstruction alone.

**Manning-Coe et al. 2026, "Interactions Between Crosscoder Features: A Compact Proofs Perspective"** ([arXiv 2606.09940](https://arxiv.org/abs/2606.09940); code <https://github.com/chainik1125/crosscoders-feature-interactions/tree/arxiv>).
- Main experiments: BatchTopK crosscoders (hidden size 1536, K = 20) on TinyStories-Instruct-33M.
- The modular-addition transformer appears as an appendix benchmark, where the authors note that each neuron hosts at least a sine and a cosine Fourier component, which limits how far a "computationally sparse" dictionary can concentrate each neuron's weight on one feature *(appendix details unverified)*.
- This is the closest prior use of sparse dictionaries on this model that we found. It does not study SAE steering.

## 3. Circular features and manifolds in SAEs

**Engels, Michaud, Liao, Gurnee, Tegmark 2024, "Not All Language Model Features Are One-Dimensionally Linear"** (ICLR 2025, [arXiv 2405.14860](https://arxiv.org/abs/2405.14860); code <https://github.com/JoshEngels/MultiDimensionalFeatures>).
- Clusters SAE decoder directions by cosine similarity (thresholded graph → connected components; spectral clustering for GPT-2) to find multi-dimensional features. This reveals circles for days of the week and months in GPT-2 and Mistral 7B.
- Interventions on these circles in Mistral 7B and Llama 3 8B show they are used for modular addition over days and months: "these circular features are indeed the fundamental unit of computation in these tasks".

*Reused:* decoder-cosine clustering for unsupervised ring recovery, and the idea of steering by moving along a circle.

**Michaud, Gorton, McGrath 2025, "Understanding sparse autoencoder scaling in the presence of feature manifolds"** ([arXiv 2509.02565](https://arxiv.org/abs/2509.02565); workshop paper). ReLU+L1 SAEs on points from a circle "can slightly reduce their total loss by 'tiling' the manifold". In one regime, feature manifolds cause SAEs "to learn far fewer features".
*Reused:* the prediction that SAE features on each key-frequency ring are localized arcs.

**Bhalla, Fel, Rager et al. 2026, "Do Sparse Autoencoders Capture Concept Manifolds?"** ([arXiv 2604.28119](https://arxiv.org/abs/2604.28119)).
- SAEs capture manifolds "globally" (a compact group of atoms spanning the manifold) or "locally" (tiling), with a mixed "dilution" regime common in practice.
- Steering between concept centroids gives smooth changes, while steering through individual features is "often ineffective or brittle".
- Communities of interacting atoms recover the manifold.

*Reused:* treating rings (groups of features), not single features, as the unit for transport steering. Also the expectation that single-feature steering may be weak, which H5 tests.

## 4. Steering, interventions and baselines

**Templeton et al. 2024, "Scaling Monosemanticity"** ([Transformer Circuits](https://transformer-circuits.pub/2024/scaling-monosemanticity/)). Feature steering by clamping a feature's activation to a multiple of its maximum training activation. The Golden Gate Bridge feature clamped to 10× its maximum made Claude 3 Sonnet self-identify as the bridge (per [secondary summaries](https://www.lesswrong.com/posts/zzmhsKx5dBpChKhry/comments-on-anthropic-s-scaling-monosemanticity); the primary page exceeded our fetcher's size limit).
*Reused:* clamping and scaling steering to multiples of a feature's maximum activation.

**Wu et al. 2025, "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders"** (ICML 2025, [arXiv 2501.17148](https://arxiv.org/abs/2501.17148); code <https://github.com/stanfordnlp/axbench>).
- "For steering, we find that prompting outperforms all existing methods, followed by finetuning. For concept detection, representation-based methods such as difference-in-means, perform the best. On both evaluations, SAEs are not competitive."
- DiffMean is w = mean(positive) − mean(negative), applied as steering h ← h + α·w.

*Reused:* DiffMean / class-conditional means as the main supervised steering baseline. A later rebuttal (Jørgensen & Hansen 2026, [arXiv 2605.31183](https://arxiv.org/abs/2605.31183)) reports SAEs closer to parity when features are selected and labelled with a supervised pipeline.

**Makelov, Lange, Nanda 2024, "Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control"** ([arXiv 2405.08366](https://arxiv.org/abs/2405.08366)). Supervised feature dictionaries for IOI achieve "excellent approximation, control, and interpretability". SAEs "are less successful than supervised features in controlling the model". Sufficiency means replacing the activation with its reconstruction; necessity means replacing it with E[a] + (a − â).
*Reused:* a supervised dictionary built from the known mechanism as an upper-bound baseline, and the sufficiency (spliced) vs error-preserving intervention variants.

**Marks et al. 2024, "Sparse Feature Circuits"** (ICLR 2025, [arXiv 2403.19647](https://arxiv.org/abs/2403.19647); code <https://github.com/saprmarks/feature-circuits>). Writes the activation as x = x̂ + ε(x) with an SAE error term kept in interventions, and reports that "removing residual stream SAE error nodes severely disrupts the model".
*Reused:* reporting both error-preserved and spliced interventions.

## 5. Libraries

- **TransformerLens 4.4.1** ([PyPI](https://pypi.org/project/transformer-lens/), released 2026-11-25). `HookedTransformer` was removed in 4.0; native models are built with `TransformerBridge.boot_native`. Hook names such as `blocks.0.hook_resid_mid` and `blocks.0.mlp.hook_post` are preserved (checked locally). `transformer_lens.model_bridge.sources.tl_legacy` converts legacy TransformerLens checkpoints.
- **SAELens 6.56.4** ([PyPI](https://pypi.org/project/sae-lens/), released 2026-11-26; [docs](https://decoderesearch.github.io/SAELens/dev/training_saes)). `SAETrainer(cfg, sae, data_provider, evaluator)` trains from any iterator of `(batch, d_in)` tensors. It supports Standard, TopK, BatchTopK and JumpReLU training configs, and BatchTopK SAEs save as JumpReLU for inference. It requires transformer-lens ≥ 3, < 5.
- **trackio** ([docs](https://huggingface.co/docs/trackio)): a local-first, wandb-compatible dashboard (`trackio show`).

## 6. Gap this project addresses

Our searches (web, arXiv, Hugging Face, LessWrong) found no peer-reviewed study that trains SAEs on the Nanda et al. modular-addition transformer and tests feature steering against the known algorithm. The only SAE-specific work found on grokked modular-arithmetic models is non-peer-reviewed. One example is a [GitHub reproduction](https://github.com/nilsleut/Grokking-Generalization-Beyond-Overfitting) that reports SAE feature counts but no steering. Because the algorithm is known exactly, this model lets us:
1. Score SAE features against ground-truth mechanism coordinates instead of LLM explanations.
2. Derive steering predictions in advance from the weights.
3. Use SAE steering to discriminate between two published algorithms (Clock vs Pizza) in this exact model.
