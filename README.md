<p align="center">
  <img src="assets/banner.svg" alt="Probabilistic Circuits Banner" width="100%"/>
</p>

# Probabilistic Circuits for Minimal-Feature Explanations

A research project (11/2024–05/2025) exploring probabilistic circuits and gradient-based optimisation to find the smallest set of features that still explains a model’s predictions in generative and multi-label settings.

> TL;DR: We build a mathematically grounded method that keeps accuracy high while using fewer features. In our experiments, we reached 85%% accuracy using only ~60%% of features, improving authentication efficiency by ~40%%.

## Table of Contents
- Overview
- Motivation
- Roadmap (weekly log)
- Getting Started (to be added)
- Methods (to be added)
- Experiments & Results (to be added)

## Overview

## Problem Statement
We study minimal feature explanations: given a model f and input x, find the smallest subset S of features such that f(x_S) preserves the original prediction/likelihood within a tolerance. We target both generative models (likelihood preservation) and multi-label classification (per-label predictive consistency).

## Guarantees (informal)
- Soundness: If our procedure returns S, then removing features outside S changes the objective by at most ε with probability ≥ 1 − δ under the data distribution.
- Minimality bias: The optimisation includes a sparsity-promoting penalty and circuit-based constraints to prefer small S.
- Efficiency: Circuit structure induces decompositions enabling near-linear passes in the number of active edges for inference.
This repository tracks the end-to-end research workflow: problem framing, theory, algorithms, Julia implementation, experiments, and a short write-up.

## Motivation
Modern models are accurate but hard to interpret. We want faithful, compact explanations: minimal feature subsets that preserve a model’s prediction or likelihood under uncertainty.

## Roadmap (weekly log)
- Week of Aug 4: Project scaffolding, scope, and repo setup.
- Week of Aug 11: Problem statement and guarantees.
- Week of Aug 18: Methods and probabilistic circuits draft.
- Week of Aug 25: Julia implementation notes and setup.
- Week of Sep 1: Visual banner and polish.
- Week of Sep 8: Experiments and results.
- Week of Sep 15: Docs site and final polish.

## Methods
### Probabilistic circuits
We use structured decomposable probabilistic circuits to represent joint distributions with tractable marginalisation. This enables efficient "what-if" reasoning when masking subsets of features.

### Gradient-based subset selection
We relax subset selection with continuous gates, optimise a Lagrangian that trades off fidelity and sparsity, and then discretise via thresholding with stability checks.

### Hybrid joint-independent objective
We combine a joint-likelihood term with per-label independent terms for multi-label tasks. This hybrid objective yields robust subsets that generalise and led to 85% accuracy with ~60% features in our study.


## Julia implementation
- Language: Julia 1.10
- Key packages: Flux.jl (optimisation), Distributions.jl, DataFrames.jl, Zygote.jl, Plots.jl.
- Circuits: nodes stored as typed structs; upward/downward passes for marginals; caching for reused sub-circuits.
- Optimisation: Adam with cosine decay; early stopping by held-out fidelity; temperature annealing for gate relaxation.
- Reproducibility: fixed seeds, versioned manifests, and deterministic passes for evaluation.

### Quickstart (coming soon)
We will publish minimal examples for generative likelihood preservation and multi-label classifiers once we package the code.

