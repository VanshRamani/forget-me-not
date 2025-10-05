# Bayesian PAC Unlearning Framework

A modular implementation of the Bayesian PAC (Probably Approximately Correct) Unlearning framework for machine learning model unlearning.

## Overview

This framework implements a principled approach to machine unlearning that:
- **Forgets** specified data (forget region X₁) 
- **Retains** model utility on remaining data (retain region X₀)
- Provides **theoretical guarantees** via PAC conditions (ε-MC and λ-UC)

## Framework Components

### 1. Synthetic Dataset Generation (`src/synthetic.py`)

Create datasets with clearly defined forget/retain regions:

```python
from src.synthetic import generate_moon_dataset, TargetConcept, partition_data

# Generate data
X, y = generate_moon_dataset(n_samples=500)

# Define forget region (e.g., spatial region)
concept = TargetConcept.spatial_region(x_min=-0.5, x_max=1.5, y_min=-0.5, y_max=0.5)

# Partition into retain/forget
X_retain, X_forget, partition = partition_data(X, y, concept)
```

**Supported datasets:**
- `generate_moon_dataset()` - Two interleaving moons
- `generate_circles_dataset()` - Concentric circles
- `generate_blobs_dataset()` - Gaussian blobs
- `generate_gaussian_mixture()` - Custom Gaussian mixtures

**Supported concepts:**
- `TargetConcept.spatial_region()` - Rectangular region
- `TargetConcept.radial()` - Circular region
- `TargetConcept.halfspace()` - Linear boundary
- `TargetConcept.class_based()` - Class-based forgetting

### 2. Bayesian Unlearning (`src/bayesian.py`)

Modular components for Bayesian inference:

#### Priors
- `DirichletProcessPrior` - Flexible Dirichlet Process prior
- `GaussianPrior` - Gaussian prior on mixture components

#### Distributions
- `GaussianMixtureDistribution` - Gaussian mixture model
- `KDEDistribution` - Kernel density estimate

#### Likelihoods
- `RetainLikelihood` - Standard likelihood: L(D₀|P) = ∏ P(x)
- `ForgetLikelihood` - Tilted likelihood: L(D₁|P) ∝ [∏ P(x)]^(-λ)
- `JointLikelihood` - Combined retain + forget

#### Main Algorithm
```python
from src.bayesian import BayesianUnlearning, GaussianPrior

# Setup prior
prior = GaussianPrior(mean_prior_mean=data_mean, mean_prior_cov=data_cov)

# Create unlearner
unlearner = BayesianUnlearning(prior=prior, lambda_hyper=2.0)

# Fit (performs Bayesian inference)
unlearner.fit(X_retain, X_forget, n_posterior_samples=50)

# Get unlearned model
Q_unlearned = unlearner.get_posterior_predictive()

# Evaluate
metrics = unlearner.evaluate(P_ideal, X_forget)
# Returns: {'epsilon': ..., 'lambda': ..., 'satisfies_pac': ...}
```

### 3. Unlearning Metrics

#### PAC Framework Metrics

**ε-Matching Condition (ε-MC):** Measures utility
- `d_TV(Q, P₀) ≤ ε` - Output Q is close to ideal distribution P₀

**λ-Unlearning Condition (λ-UC):** Measures forgetting
- `Q(X₁) ≤ λ` - Low probability mass in forget region

**Key Theorem:** ε-MC implies λ-UC with λ = ε

#### Additional Standard Unlearning Metrics

The framework also computes commonly used distribution distance metrics:

**Information-Theoretic:**
- **KL Divergence** `D_KL(P₀||Q)` - Asymmetric measure of how P₀ diverges from Q
- **JS Divergence** - Symmetric version of KL, bounded in [0, log(2)] ≈ [0, 0.693]

**Geometric Distances:**
- **Hellinger Distance** - Symmetric, bounded in [0, 1]
- **Wasserstein Distance (W₁)** - Earth Mover's Distance (sliced for multivariate)
- **Chi-Squared Distance** - Weighted difference measure

All metrics are computed automatically during evaluation and can be used to comprehensively assess unlearning quality.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install numpy scipy scikit-learn matplotlib
```

## Usage

### Quick Start

Run the demo script:

```bash
python demo.py
```

This will:
1. Generate synthetic dataset with forget/retain regions
2. Fit ideal retain distribution P₀
3. Perform Bayesian unlearning with different λ_hyper values
4. Evaluate PAC metrics (ε and λ)
5. Visualize results

### Custom Experiment

```python
from src.synthetic import generate_moon_dataset, TargetConcept, partition_data
from src.bayesian import BayesianUnlearning, GaussianPrior, fit_distribution_to_data

# 1. Generate data
X, y = generate_moon_dataset(n_samples=800)
concept = TargetConcept.radial(center=(1.0, 0.0), radius=0.7)
X_retain, X_forget, _ = partition_data(X, y, concept)

# 2. Fit ideal retain distribution
P_ideal = fit_distribution_to_data(X_retain, method='gmm', n_components=5)

# 3. Setup and run unlearning
prior = GaussianPrior(mean_prior_mean=X_retain.mean(axis=0), 
                     mean_prior_cov=np.cov(X_retain.T))
unlearner = BayesianUnlearning(prior=prior, lambda_hyper=2.0)
unlearner.fit(X_retain, X_forget, n_posterior_samples=50)

# 4. Get results
Q_unlearned = unlearner.get_posterior_predictive()
metrics = unlearner.evaluate(P_ideal, X_forget)
print(f"ε = {metrics['epsilon']:.4f}, λ = {metrics['lambda']:.4f}")
```

## Modular Design

The framework is designed for easy customization:

### Change Prior
```python
# Instead of GaussianPrior, use DirichletProcess
from src.bayesian import DirichletProcessPrior, GaussianMixtureDistribution

base_measure = fit_distribution_to_data(X_retain, method='gmm')
prior = DirichletProcessPrior(concentration=1.0, base_measure=base_measure)
```

### Change Likelihood
```python
# Customize forgetting strength
from src.bayesian import ForgetLikelihood

forget_lik = ForgetLikelihood(lambda_hyper=5.0)  # Stronger forgetting
```

### Change Distribution Representation
```python
# Use KDE instead of GMM
from src.bayesian import KDEDistribution

P_ideal = KDEDistribution(X_retain)
```

### Change Target Concept
```python
# Custom concept function
def my_concept(X):
    # Forget points where x1 > x2
    return (X[:, 0] > X[:, 1]).astype(int)

concept = TargetConcept(my_concept, name="custom")
```

## Theory

### PAC Unlearning Framework

Given:
- Instance space X
- Target concept h: X → {0,1} (0=retain, 1=forget)
- Ideal retain distribution P₀
- Output distribution Q (from unlearning algorithm)

Objectives:
1. **ε-MC:** d(Q, P₀) ≤ ε (maintain utility)
2. **λ-UC:** Q(X₁) ≤ λ (achieve forgetting)

### Bayesian Formulation

Posterior inference:
```
π(P | D₀, D₁) ∝ L(D₀|P) × L(D₁|P) × π(P)
                ∝ [∏ P(x)]      × [∏ P(x)]^(-λ_hyper) × π(P)
                   x∈D₀           x∈D₁
```

Output: Q_Bayes(x) = 𝔼[P(x) | D₀, D₁]

## File Structure

```
PAC-Bayesian/
├── src/
│   ├── synthetic.py      # Dataset generation
│   └── bayesian.py       # Bayesian unlearning framework
├── demo.py               # Demonstration script
├── requirements.txt      # Dependencies
├── Framework.md          # Theoretical framework
└── README.md            # This file
```

## Key Hyperparameters

- **λ_hyper**: Forgetting strength (higher = stronger forgetting, but may reduce utility)
- **n_posterior_samples**: Number of samples for posterior approximation
- **n_components**: Number of mixture components for distributions
- **concentration** (DP prior): Controls prior flexibility

## Examples

See `demo.py` for:
- Basic unlearning experiment
- Comparison of different λ_hyper values
- Visualization of results

## License

MIT License

## Contributing

The modular design encourages experimentation. Feel free to:
- Add new prior distributions
- Implement alternative likelihood functions
- Create new target concepts
- Extend to higher dimensions or different data types
