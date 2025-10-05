"""
Bayesian PAC Unlearning Framework Implementation.

This module implements the Bayesian PAC Unlearning framework with modular components:
- Prior distributions (Dirichlet Process, Gaussian priors)
- Likelihood functions (retain and forget)
- Posterior inference
- PAC metrics (ε-MC, λ-UC, TV distance)
"""

import numpy as np
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.spatial.distance import cdist
from typing import Callable, Optional, Tuple, Dict, List
from abc import ABC, abstractmethod
import warnings


# ==================== Prior Distributions ====================

class Prior(ABC):
    """Abstract base class for prior distributions over probability distributions."""
    
    @abstractmethod
    def sample(self, n_components: int = 10) -> 'Distribution':
        """Sample a distribution from the prior."""
        pass
    
    @abstractmethod
    def log_prob(self, distribution: 'Distribution') -> float:
        """Compute log probability of a distribution under the prior."""
        pass


class DirichletProcessPrior(Prior):
    """
    Dirichlet Process prior: P ~ DP(α, H)
    
    Approximated using a stick-breaking construction with truncation.
    """
    
    def __init__(self, concentration: float, base_measure: 'Distribution'):
        """
        Args:
            concentration (α): Concentration parameter controlling flexibility
            base_measure (H): Base distribution (prior guess)
        """
        self.concentration = concentration
        self.base_measure = base_measure
    
    def sample(self, n_components: int = 10, random_state: Optional[int] = None) -> 'Distribution':
        """Sample from DP using stick-breaking construction."""
        rng = np.random.RandomState(random_state)
        
        # Stick-breaking process for weights
        betas = rng.beta(1, self.concentration, size=n_components)
        weights = np.zeros(n_components)
        weights[0] = betas[0]
        for k in range(1, n_components):
            weights[k] = betas[k] * np.prod(1 - betas[:k])
        weights = weights / weights.sum()  # Normalize
        
        # Sample component parameters from base measure
        components = []
        for _ in range(n_components):
            # Sample from base measure (simplified: perturb base measure)
            if hasattr(self.base_measure, 'means'):
                mean = self.base_measure.means[0] + rng.randn(self.base_measure.dim) * 0.5
                cov = self.base_measure.covs[0] * (0.5 + rng.rand())
            else:
                mean = rng.randn(2) * 2
                cov = np.eye(2) * 0.5
            components.append({'mean': mean, 'cov': cov})
        
        return GaussianMixtureDistribution(weights, components)
    
    def log_prob(self, distribution: 'Distribution') -> float:
        """Compute log-probability (simplified)."""
        # Simplified: penalize complexity
        if hasattr(distribution, 'weights'):
            n_components = len(distribution.weights)
            return -0.5 * n_components / self.concentration
        return 0.0


class GaussianPrior(Prior):
    """
    Simple Gaussian prior for mixture components.
    Each component has a Gaussian prior on mean and Wishart on covariance.
    """
    
    def __init__(self, mean_prior_mean: np.ndarray, mean_prior_cov: np.ndarray,
                 dim: int = 2, n_components: int = 5):
        """
        Args:
            mean_prior_mean: Prior mean for component means
            mean_prior_cov: Prior covariance for component means
            dim: Dimensionality of data
            n_components: Number of mixture components
        """
        self.mean_prior_mean = mean_prior_mean
        self.mean_prior_cov = mean_prior_cov
        self.dim = dim
        self.n_components = n_components
    
    def sample(self, n_components: Optional[int] = None, random_state: Optional[int] = None) -> 'Distribution':
        """Sample from Gaussian prior."""
        rng = np.random.RandomState(random_state)
        n_comp = n_components or self.n_components
        
        # Sample weights from Dirichlet
        weights = rng.dirichlet(np.ones(n_comp))
        
        # Sample component parameters
        components = []
        for _ in range(n_comp):
            mean = rng.multivariate_normal(self.mean_prior_mean, self.mean_prior_cov)
            cov = np.eye(self.dim) * (0.3 + rng.rand() * 0.5)  # Simplified
            components.append({'mean': mean, 'cov': cov})
        
        return GaussianMixtureDistribution(weights, components)
    
    def log_prob(self, distribution: 'Distribution') -> float:
        """Compute log-probability of distribution."""
        if not hasattr(distribution, 'means'):
            return -np.inf
        
        log_prob = 0.0
        for mean in distribution.means:
            log_prob += multivariate_normal.logpdf(mean, self.mean_prior_mean, self.mean_prior_cov)
        return log_prob


# ==================== Distribution Representations ====================

class Distribution(ABC):
    """Abstract base class for probability distributions."""
    
    @abstractmethod
    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Compute probability density at points X."""
        pass
    
    @abstractmethod
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the distribution."""
        pass
    
    def log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of data X."""
        densities = self.pdf(X)
        densities = np.maximum(densities, 1e-300)  # Avoid log(0)
        return np.sum(np.log(densities))


class GaussianMixtureDistribution(Distribution):
    """Gaussian Mixture Model distribution."""
    
    def __init__(self, weights: np.ndarray, components: List[Dict]):
        """
        Args:
            weights: Mixture weights of shape (n_components,)
            components: List of dicts with 'mean' and 'cov' keys
        """
        self.weights = weights
        self.components = components
        self.n_components = len(weights)
        self.means = np.array([c['mean'] for c in components])
        self.covs = [c['cov'] for c in components]
        self.dim = len(components[0]['mean'])
    
    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Compute probability density."""
        n_samples = X.shape[0]
        density = np.zeros(n_samples)
        
        for k in range(self.n_components):
            component_density = multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covs[k]
            )
            density += self.weights[k] * component_density
        
        return density
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from the mixture."""
        rng = np.random.RandomState(random_state)
        
        # Sample component assignments
        components = rng.choice(self.n_components, size=n_samples, p=self.weights)
        
        # Sample from each component
        samples = np.zeros((n_samples, self.dim))
        for i in range(n_samples):
            k = components[i]
            samples[i] = rng.multivariate_normal(self.means[k], self.covs[k])
        
        return samples


class KDEDistribution(Distribution):
    """Kernel Density Estimate distribution."""
    
    def __init__(self, data: np.ndarray, bandwidth: Optional[float] = None):
        """
        Args:
            data: Training data of shape (n_samples, n_features)
            bandwidth: KDE bandwidth (if None, uses Scott's rule)
        """
        self.data = data
        self.dim = data.shape[1]
        
        if bandwidth is not None:
            self.kde = gaussian_kde(data.T, bw_method=bandwidth)
        else:
            self.kde = gaussian_kde(data.T)
    
    def pdf(self, X: np.ndarray) -> np.ndarray:
        """Compute probability density."""
        return self.kde(X.T)
    
    def sample(self, n_samples: int, random_state: Optional[int] = None) -> np.ndarray:
        """Sample from KDE."""
        if random_state is not None:
            np.random.seed(random_state)
        return self.kde.resample(n_samples).T


# ==================== Likelihood Functions ====================

class LikelihoodFunction(ABC):
    """Abstract base class for likelihood functions."""
    
    @abstractmethod
    def compute(self, data: np.ndarray, distribution: Distribution) -> float:
        """Compute likelihood of data under distribution."""
        pass


class RetainLikelihood(LikelihoodFunction):
    """
    Standard likelihood for retain data: L(D₀ | P) = ∏_{x ∈ D₀} P(x)
    """
    
    def compute(self, data: np.ndarray, distribution: Distribution) -> float:
        """Compute retain likelihood (log scale)."""
        return distribution.log_likelihood(data)


class ForgetLikelihood(LikelihoodFunction):
    """
    Tilted likelihood for forget data: L(D₁ | P) ∝ [∏_{x ∈ D₁} P(x)]^(-λ_hyper)
    
    This penalizes distributions that assign high probability to forget data.
    """
    
    def __init__(self, lambda_hyper: float = 1.0):
        """
        Args:
            lambda_hyper: Forgetting strength (higher = stronger forgetting)
        """
        self.lambda_hyper = lambda_hyper
    
    def compute(self, data: np.ndarray, distribution: Distribution) -> float:
        """Compute forget likelihood (log scale)."""
        if len(data) == 0:
            return 0.0
        log_lik = distribution.log_likelihood(data)
        return -self.lambda_hyper * log_lik


class JointLikelihood(LikelihoodFunction):
    """
    Combined likelihood: L(D₀, D₁ | P) = L(D₀ | P) * L(D₁ | P)
    """
    
    def __init__(self, retain_likelihood: RetainLikelihood, 
                 forget_likelihood: ForgetLikelihood):
        self.retain_likelihood = retain_likelihood
        self.forget_likelihood = forget_likelihood
    
    def compute(self, retain_data: np.ndarray, forget_data: np.ndarray,
                distribution: Distribution) -> float:
        """Compute joint likelihood (log scale)."""
        log_lik_retain = self.retain_likelihood.compute(retain_data, distribution)
        log_lik_forget = self.forget_likelihood.compute(forget_data, distribution)
        return log_lik_retain + log_lik_forget


# ==================== PAC Metrics ====================

def total_variation_distance(P: Distribution, Q: Distribution, 
                             n_samples: int = 10000, 
                             random_state: Optional[int] = 42) -> float:
    """
    Estimate Total Variation distance: d_TV(P, Q) = sup_A |P(A) - Q(A)|
    
    Approximated using Monte Carlo sampling.
    
    Args:
        P, Q: Two distributions
        n_samples: Number of samples for approximation
        random_state: Random seed
    
    Returns:
        Estimated TV distance
    """
    rng = np.random.RandomState(random_state)
    
    # Sample from both distributions
    samples_P = P.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    samples_Q = Q.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    all_samples = np.vstack([samples_P, samples_Q])
    
    # Compute densities
    p_vals = P.pdf(all_samples)
    q_vals = Q.pdf(all_samples)
    
    # TV distance approximation
    tv_distance = 0.5 * np.mean(np.abs(p_vals - q_vals))
    
    return tv_distance


def kl_divergence(P: Distribution, Q: Distribution,
                 n_samples: int = 10000,
                 random_state: Optional[int] = 42) -> float:
    """
    Estimate KL divergence: D_KL(P || Q) = E_P[log(P(x) / Q(x))]
    
    Measures how P diverges from Q (asymmetric).
    
    Args:
        P: Reference distribution
        Q: Approximating distribution
        n_samples: Number of samples for Monte Carlo estimation
        random_state: Random seed
    
    Returns:
        Estimated KL divergence (in nats)
    """
    # Sample from P
    samples = P.sample(n_samples, random_state=random_state)
    
    # Compute densities
    p_vals = P.pdf(samples)
    q_vals = Q.pdf(samples)
    
    # Avoid numerical issues
    p_vals = np.maximum(p_vals, 1e-300)
    q_vals = np.maximum(q_vals, 1e-300)
    
    # KL divergence: E[log(P/Q)]
    kl_div = np.mean(np.log(p_vals) - np.log(q_vals))
    
    return max(0.0, kl_div)  # KL is non-negative


def js_divergence(P: Distribution, Q: Distribution,
                 n_samples: int = 10000,
                 random_state: Optional[int] = 42) -> float:
    """
    Estimate Jensen-Shannon divergence: JS(P, Q) = 0.5 * [KL(P||M) + KL(Q||M)]
    where M = 0.5 * (P + Q)
    
    Symmetric measure bounded in [0, log(2)] ≈ [0, 0.693].
    
    Args:
        P, Q: Two distributions
        n_samples: Number of samples for estimation
        random_state: Random seed
    
    Returns:
        Estimated JS divergence (in nats)
    """
    rng = np.random.RandomState(random_state)
    
    # Sample from both distributions
    samples_P = P.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    samples_Q = Q.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    all_samples = np.vstack([samples_P, samples_Q])
    
    # Compute densities
    p_vals = P.pdf(all_samples)
    q_vals = Q.pdf(all_samples)
    m_vals = 0.5 * (p_vals + q_vals)
    
    # Avoid numerical issues
    p_vals = np.maximum(p_vals, 1e-300)
    q_vals = np.maximum(q_vals, 1e-300)
    m_vals = np.maximum(m_vals, 1e-300)
    
    # JS divergence
    js_div = 0.5 * (np.mean(np.log(p_vals) - np.log(m_vals)) + 
                    np.mean(np.log(q_vals) - np.log(m_vals)))
    
    return max(0.0, js_div)


def hellinger_distance(P: Distribution, Q: Distribution,
                      n_samples: int = 10000,
                      random_state: Optional[int] = 42) -> float:
    """
    Estimate Hellinger distance: H(P, Q) = sqrt(1 - ∫sqrt(P(x)Q(x))dx)
    
    Symmetric measure bounded in [0, 1].
    
    Args:
        P, Q: Two distributions
        n_samples: Number of samples for estimation
        random_state: Random seed
    
    Returns:
        Estimated Hellinger distance
    """
    rng = np.random.RandomState(random_state)
    
    # Sample from both distributions
    samples_P = P.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    samples_Q = Q.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    all_samples = np.vstack([samples_P, samples_Q])
    
    # Compute densities
    p_vals = P.pdf(all_samples)
    q_vals = Q.pdf(all_samples)
    
    # Hellinger distance
    bc_coefficient = np.mean(np.sqrt(p_vals * q_vals))  # Bhattacharyya coefficient
    hellinger = np.sqrt(max(0.0, 1.0 - bc_coefficient))
    
    return hellinger


def wasserstein_distance(P: Distribution, Q: Distribution,
                        n_samples: int = 5000,
                        random_state: Optional[int] = 42) -> float:
    """
    Estimate 1-Wasserstein (Earth Mover's) distance using samples.
    
    W_1(P, Q) = inf E[||X - Y||] over all couplings of P and Q.
    Approximated using sample-based optimal transport.
    
    Args:
        P, Q: Two distributions
        n_samples: Number of samples (smaller due to O(n²) complexity)
        random_state: Random seed
    
    Returns:
        Estimated Wasserstein-1 distance
    """
    from scipy.stats import wasserstein_distance as wd_1d
    
    # Sample from both
    samples_P = P.sample(n_samples, random_state=random_state)
    samples_Q = Q.sample(n_samples, random_state=random_state + 1)
    
    # For multivariate case, use average over dimensions (approximation)
    # Better: use POT library, but avoiding extra dependency
    if samples_P.shape[1] == 1:
        return wd_1d(samples_P.flatten(), samples_Q.flatten())
    else:
        # Sliced Wasserstein: average over random projections
        n_projections = 50
        rng = np.random.RandomState(random_state)
        distances = []
        
        for _ in range(n_projections):
            # Random unit vector
            direction = rng.randn(samples_P.shape[1])
            direction = direction / np.linalg.norm(direction)
            
            # Project samples
            proj_P = samples_P @ direction
            proj_Q = samples_Q @ direction
            
            # 1D Wasserstein
            distances.append(wd_1d(proj_P, proj_Q))
        
        return np.mean(distances)


def chi_squared_distance(P: Distribution, Q: Distribution,
                        n_samples: int = 10000,
                        random_state: Optional[int] = 42) -> float:
    """
    Estimate Chi-squared distance: χ²(P, Q) = ∫(P(x) - Q(x))² / Q(x) dx
    
    Asymmetric measure (use Q as reference).
    
    Args:
        P, Q: Two distributions
        n_samples: Number of samples for estimation
        random_state: Random seed
    
    Returns:
        Estimated χ² distance
    """
    # Sample from mixture of P and Q
    rng = np.random.RandomState(random_state)
    samples_P = P.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    samples_Q = Q.sample(n_samples // 2, random_state=rng.randint(0, 10000))
    all_samples = np.vstack([samples_P, samples_Q])
    
    # Compute densities
    p_vals = P.pdf(all_samples)
    q_vals = Q.pdf(all_samples)
    q_vals = np.maximum(q_vals, 1e-300)  # Avoid division by zero
    
    # χ² distance
    chi2 = np.mean((p_vals - q_vals)**2 / q_vals)
    
    return max(0.0, chi2)


def unlearning_condition(Q: Distribution, forget_region_data: np.ndarray,
                        n_samples: int = 10000, random_state: Optional[int] = 42) -> float:
    """
    Compute λ-UC: Q(X₁) = probability mass in forget region.
    
    Approximated by sampling from Q and checking membership in forget region.
    
    Args:
        Q: Output distribution
        forget_region_data: Sample points from the forget region (to define it)
        n_samples: Number of samples from Q
        random_state: Random seed
    
    Returns:
        λ value (probability mass in forget region)
    """
    samples = Q.sample(n_samples, random_state=random_state)
    
    # Estimate forget region membership using KDE on forget data
    if len(forget_region_data) > 0:
        kde_forget = gaussian_kde(forget_region_data.T)
        # Points with high density under forget KDE are likely in forget region
        forget_densities = kde_forget(samples.T)
        threshold = np.percentile(kde_forget(forget_region_data.T), 25)  # Conservative
        in_forget_region = forget_densities > threshold
        lambda_uc = np.mean(in_forget_region)
    else:
        lambda_uc = 0.0
    
    return lambda_uc


# ==================== Bayesian Unlearning Algorithm ====================

class BayesianUnlearning:
    """
    Main class implementing the Bayesian PAC Unlearning framework.
    
    Performs Bayesian inference: π(P | D₀, D₁) ∝ L(D₀ | P) * L(D₁ | P) * π(P)
    """
    
    def __init__(self, prior: Prior, lambda_hyper: float = 1.0):
        """
        Args:
            prior: Prior distribution over distributions P
            lambda_hyper: Forgetting strength hyperparameter
        """
        self.prior = prior
        self.lambda_hyper = lambda_hyper
        self.retain_likelihood = RetainLikelihood()
        self.forget_likelihood = ForgetLikelihood(lambda_hyper)
        self.posterior_samples = []
        self.posterior_weights = []
    
    def fit(self, D_retain: np.ndarray, D_forget: np.ndarray,
            n_posterior_samples: int = 100, n_components: int = 5,
            random_state: Optional[int] = 42) -> None:
        """
        Perform posterior inference using importance sampling.
        
        Args:
            D_retain: Retain data (n_retain, n_features)
            D_forget: Forget data (n_forget, n_features)
            n_posterior_samples: Number of samples from posterior
            n_components: Number of components for mixture models
            random_state: Random seed
        """
        rng = np.random.RandomState(random_state)
        
        print(f"Fitting Bayesian Unlearning with λ_hyper={self.lambda_hyper}")
        print(f"Retain data: {len(D_retain)} samples")
        print(f"Forget data: {len(D_forget)} samples")
        
        # Importance sampling from prior
        log_weights = []
        samples = []
        
        for i in range(n_posterior_samples):
            # Sample from prior
            P_sample = self.prior.sample(n_components=n_components, 
                                        random_state=rng.randint(0, 1000000))
            
            # Compute log-likelihood
            log_lik_retain = self.retain_likelihood.compute(D_retain, P_sample)
            log_lik_forget = self.forget_likelihood.compute(D_forget, P_sample)
            log_lik = log_lik_retain + log_lik_forget
            
            # Store
            samples.append(P_sample)
            log_weights.append(log_lik)
            
            if (i + 1) % 20 == 0:
                print(f"  Sampled {i + 1}/{n_posterior_samples} distributions")
        
        # Normalize weights
        log_weights = np.array(log_weights)
        log_weights = log_weights - np.max(log_weights)  # Numerical stability
        weights = np.exp(log_weights)
        weights = weights / weights.sum()
        
        self.posterior_samples = samples
        self.posterior_weights = weights
        
        print(f"Posterior inference complete. Effective sample size: {1 / np.sum(weights**2):.1f}")
    
    def get_posterior_predictive(self) -> Distribution:
        """
        Compute posterior predictive distribution: Q_Bayes(x) = E_P[P(x)]
        
        Returns:
            Posterior predictive as a mixture distribution
        """
        if not self.posterior_samples:
            raise ValueError("Must call fit() first")
        
        # Create mixture of all posterior samples weighted by posterior weights
        all_means = []
        all_covs = []
        all_weights = []
        
        for P, w in zip(self.posterior_samples, self.posterior_weights):
            if hasattr(P, 'means'):
                for k in range(len(P.means)):
                    all_means.append(P.means[k])
                    all_covs.append(P.covs[k])
                    all_weights.append(w * P.weights[k])
        
        all_means = np.array(all_means)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
        
        components = [{'mean': all_means[i], 'cov': all_covs[i]} 
                     for i in range(len(all_means))]
        
        return GaussianMixtureDistribution(all_weights, components)
    
    def evaluate(self, P_ideal: Distribution, forget_region_data: np.ndarray,
                n_samples: int = 10000, compute_all_metrics: bool = True) -> Dict[str, float]:
        """
        Evaluate the unlearned model using PAC metrics and standard unlearning metrics.
        
        Args:
            P_ideal: Ideal retain distribution P₀
            forget_region_data: Samples from forget region
            n_samples: Number of samples for metric computation
            compute_all_metrics: If True, compute additional distance metrics beyond ε and λ
        
        Returns:
            Dictionary with metrics:
                - epsilon: TV distance (PAC ε-MC)
                - lambda: Forget region mass (PAC λ-UC)
                - satisfies_pac: Boolean indicating PAC conditions
                - kl_divergence: KL(P₀ || Q)
                - js_divergence: Jensen-Shannon divergence
                - hellinger: Hellinger distance
                - wasserstein: Wasserstein-1 distance
                - chi_squared: Chi-squared distance
        """
        Q = self.get_posterior_predictive()
        
        # Core PAC metrics
        epsilon = total_variation_distance(Q, P_ideal, n_samples=n_samples)
        lambda_uc = unlearning_condition(Q, forget_region_data, n_samples=n_samples)
        
        results = {
            'epsilon': epsilon,
            'lambda': lambda_uc,
            'satisfies_pac': epsilon <= 0.5 and lambda_uc <= epsilon
        }
        
        # Additional standard unlearning metrics
        if compute_all_metrics:
            print("  Computing additional distance metrics...")
            
            # KL divergence (asymmetric: how P_ideal differs from Q)
            results['kl_divergence'] = kl_divergence(P_ideal, Q, n_samples=n_samples)
            
            # JS divergence (symmetric)
            results['js_divergence'] = js_divergence(P_ideal, Q, n_samples=n_samples)
            
            # Hellinger distance (symmetric, bounded [0,1])
            results['hellinger'] = hellinger_distance(P_ideal, Q, n_samples=n_samples)
            
            # Wasserstein distance (Earth Mover's Distance)
            results['wasserstein'] = wasserstein_distance(P_ideal, Q, n_samples=min(5000, n_samples))
            
            # Chi-squared distance
            results['chi_squared'] = chi_squared_distance(P_ideal, Q, n_samples=n_samples)
        
        return results


# ==================== Utility Functions ====================

def fit_distribution_to_data(data: np.ndarray, 
                            method: str = 'gmm',
                            n_components: int = 5,
                            random_state: Optional[int] = 42) -> Distribution:
    """
    Fit a distribution to data.
    
    Args:
        data: Training data
        method: 'gmm' for Gaussian Mixture or 'kde' for Kernel Density
        n_components: Number of components (for GMM)
        random_state: Random seed
    
    Returns:
        Fitted distribution
    """
    if method == 'kde':
        return KDEDistribution(data)
    elif method == 'gmm':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(data)
        
        components = [{'mean': gmm.means_[k], 'cov': gmm.covariances_[k]}
                     for k in range(n_components)]
        return GaussianMixtureDistribution(gmm.weights_, components)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("Bayesian PAC Unlearning Framework")
    print("=" * 60)
    print("This module provides modular components for machine unlearning.")
    print("\nKey components:")
    print("  - Priors: DirichletProcessPrior, GaussianPrior")
    print("  - Distributions: GaussianMixtureDistribution, KDEDistribution")
    print("  - Likelihoods: RetainLikelihood, ForgetLikelihood")
    print("  - Main class: BayesianUnlearning")
    print("\nMetrics:")
    print("  PAC Framework:")
    print("    - total_variation_distance (ε-MC)")
    print("    - unlearning_condition (λ-UC)")
    print("  Information-Theoretic:")
    print("    - kl_divergence")
    print("    - js_divergence")
    print("  Geometric:")
    print("    - hellinger_distance")
    print("    - wasserstein_distance")
    print("    - chi_squared_distance")
    print("\nSee demo.py for usage examples.")
