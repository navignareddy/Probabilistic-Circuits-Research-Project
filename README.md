<p align="center">
  <img src="assets/banner.svg" alt="Probabilistic Circuits Banner" width="100%"/>
</p>

# Probabilistic Circuits for Minimal‑Feature Explanations

## Abstract
We present a principled framework that identifies compact, human‑readable feature subsets that preserve model behaviour. Our approach combines structured probabilistic circuits for tractable inference with a continuous relaxation of subset selection optimised by gradients. Across authentication‑style and synthetic benchmarks, the method achieves 85% prediction accuracy using roughly 60% of the features, improving end‑to‑end authentication throughput by about 40% due to reduced I/O and compute.

## Contributions
- Formalisation of minimal‑feature explanations for both generative likelihood and multi‑label prediction consistency.
- Differentiable subset selection with stability‑aware sparsity, enabling efficient optimisation in high‑dimensional spaces.
- Use of decomposable probabilistic circuits to guarantee tractable marginals and near‑linear inference in the number of active edges.
- Empirical study demonstrating competitive fidelity with substantially fewer features and improved system‑level efficiency.

## Problem formulation
Given an input x ∈ ℝ^d and a model f, we seek the smallest subset S ⊆ [d] that preserves model behaviour.
- Generative setting: Let p_θ(x) denote a tractable density represented by a probabilistic circuit. We define a fidelity functional F_gen(x, S) = log p_θ(x_S, x_{\bar S} marginalized), i.e., the likelihood of x when features outside S are integrated out using the circuit’s structure.
- Multi‑label setting: Let f(x) ∈ [0,1]^L denote per‑label probabilities. We require labelwise consistency, e.g., argmax_ℓ f_ℓ(x) = argmax_ℓ f_ℓ(x_S) and a small drop in calibrated scores.

The optimisation target is
\[ \min_{S \subseteq [d]} |S| \quad \text{s.t.} \quad F(x, S) \ge F(x, [d]) - \varepsilon, \]
with F instantiated as F_gen or a prediction‑consistency surrogate.

## Probabilistic circuits
A probabilistic circuit is a DAG with sum and product nodes that encodes a distribution with structural decomposability and determinism. These properties yield tractable exact marginalisation and conditionals via upward/downward passes. For masked inputs, the circuit naturally integrates out missing variables, giving faithful likelihoods for partially observed x_S. This capability is central when scoring subsets.

## Method
### Continuous gating
We relax subset selection by introducing gates z ∈ [0,1]^d applied elementwise to features: x_S ≈ z ⊙ x. Gates are initialised dense and driven to sparsity.

### Objective
We minimise a Lagrangian combining fidelity with parsimony and stability:
\[ \mathcal{L}(z) = \underbrace{\Phi(x, z)}_{\text{fidelity}} + \lambda_1 \|z\|_1 + \lambda_2 \Omega_{\text{stab}}(z), \]
where Φ is −F(x, S(z)) for generative likelihood or a calibrated consistency loss for classifiers, and Ω_stab penalises variance of gate selections across bootstrap resamples. Circuit‑level masks ensure valid factorisations when gates zero out feature groups.

### Optimisation
We use Adam with cosine decay and temperature annealing for the gate squashing function. After convergence, we threshold z to obtain a discrete S, followed by a short discrete refinement (forward‑backward swap search) to remove redundant features and re‑add critical ones if needed.

### Complexity
Inference with circuits scales with the number of active edges; one optimisation step costs O(|E_active| + d). In practice this enables hundreds of gradient steps on medium‑size datasets within minutes on CPU.

## Theoretical properties (informal)
- Soundness under tolerance: With high probability over the data distribution, removing features outside the returned S changes the objective by at most ε.
- Minimality bias: The combination of L1 penalty and circuit constraints yields a preference for small S; the discrete refinement cannot increase |S|.
- Stability: Bootstrap‑averaged gates bound selection variance, improving reproducibility across splits and seeds.

## Experimental setup
- Datasets: multivariate authentication‑like logs and synthetic mixtures with controlled sparsity.
- Baselines: greedy forward selection, L1‑regularised linear models, and gradient‑saliency with top‑k masking.
- Metrics: fidelity drop (likelihood/prediction), subset size, runtime, and stability (intersection‑over‑union across resamples).

## Results
- Hybrid joint‑independent objective attains 85% accuracy using ~60% of features on the authentication task.
- Throughput improves by ~40% because fewer features are read, transmitted, and processed at inference time.
- The selected subsets are consistent across seeds and robust to moderate distribution shift in held‑out weeks.

## Implementation notes (Julia)
- Julia 1.10; key packages: Flux.jl, Distributions.jl, DataFrames.jl, Zygote.jl, Plots.jl.
- Circuits are typed structs with cached upward/downward passes for marginals; masking hooks integrate out gated variables exactly.
- Reproducibility: fixed seeds, deterministic passes, and versioned environments.

## Citation
If this work is useful, please cite:

```
@misc{prob_circuits_minimal_features_2025,
  title        = {Probabilistic Circuits for Minimal-Feature Explanations},
  author       = {N. Reddy},
  year         = {2025},
  note         = {GitHub repository},
  url          = {https://github.com/navignareddy/Probabilistic-Circuits-Research-Project}
}
```

## Contact
Questions and collaborations: please open an issue in the repository.
