"""
Baseline unlearning methods for comparison.

This module implements standard baselines to evaluate unlearning algorithms:
1. Exact Unlearning (Retrain from Scratch) - Gold standard
2. Original Model (No Unlearning) - Trained on all data
3. Naive Fine-tuning - Fine-tune on retain data only
"""

import numpy as np
from typing import Tuple, Dict
from .bayesian import (
    Distribution, GaussianMixtureDistribution, KDEDistribution,
    fit_distribution_to_data, total_variation_distance, kl_divergence,
    js_divergence, hellinger_distance, wasserstein_distance, unlearning_condition
)


class BaselineUnlearner:
    """Base class for baseline unlearning methods."""
    
    def __init__(self, method: str = 'gmm', n_components: int = 5):
        """
        Args:
            method: Distribution fitting method ('gmm' or 'kde')
            n_components: Number of components for GMM
        """
        self.method = method
        self.n_components = n_components
        self.model = None
    
    def fit(self, X_retain: np.ndarray, X_forget: np.ndarray = None) -> None:
        """Fit the baseline model."""
        raise NotImplementedError
    
    def get_distribution(self) -> Distribution:
        """Get the learned distribution."""
        return self.model


class ExactUnlearning(BaselineUnlearner):
    """
    Exact Unlearning (Retrain from Scratch).
    
    The gold standard: train only on retain data, never seeing forget data.
    This is the ideal unlearning result (P₀ in the PAC framework).
    """
    
    def fit(self, X_retain: np.ndarray, X_forget: np.ndarray = None) -> None:
        """
        Train model only on retain data.
        
        Args:
            X_retain: Retain data
            X_forget: Ignored (model never sees this)
        """
        print(f"  [Exact Unlearning] Training from scratch on {len(X_retain)} retain samples...")
        self.model = fit_distribution_to_data(
            X_retain, 
            method=self.method, 
            n_components=self.n_components
        )
        print(f"  [Exact Unlearning] ✓ Model trained (never saw forget data)")


class OriginalModel(BaselineUnlearner):
    """
    Original Model (No Unlearning).
    
    Train on all data (retain + forget) without any unlearning.
    This represents the worst-case scenario where unlearning completely fails.
    """
    
    def fit(self, X_retain: np.ndarray, X_forget: np.ndarray) -> None:
        """
        Train model on all data (retain + forget).
        
        Args:
            X_retain: Retain data
            X_forget: Forget data
        """
        X_all = np.vstack([X_retain, X_forget])
        print(f"  [Original Model] Training on ALL data ({len(X_all)} samples: {len(X_retain)} retain + {len(X_forget)} forget)...")
        self.model = fit_distribution_to_data(
            X_all,
            method=self.method,
            n_components=self.n_components
        )
        print(f"  [Original Model] ✓ Model trained on all data (no unlearning)")


class NaiveFinetuning(BaselineUnlearner):
    """
    Naive Fine-tuning Baseline.
    
    Start with the original model (trained on all data), then fine-tune
    only on retain data. This is a simple unlearning approach that may
    not fully remove the influence of forget data.
    """
    
    def __init__(self, method: str = 'gmm', n_components: int = 5, 
                 finetune_strength: float = 0.5):
        """
        Args:
            method: Distribution fitting method
            n_components: Number of components
            finetune_strength: Weight for mixing original and retrained (0=original, 1=fully retrained)
        """
        super().__init__(method, n_components)
        self.finetune_strength = finetune_strength
        self.original_model = None
        self.retrained_model = None
    
    def fit(self, X_retain: np.ndarray, X_forget: np.ndarray) -> None:
        """
        Train original model, then fine-tune on retain data.
        
        Args:
            X_retain: Retain data
            X_forget: Forget data
        """
        # Train original model on all data
        X_all = np.vstack([X_retain, X_forget])
        print(f"  [Naive Fine-tuning] Step 1: Train original model on all {len(X_all)} samples...")
        self.original_model = fit_distribution_to_data(
            X_all,
            method=self.method,
            n_components=self.n_components
        )
        
        # Retrain on retain data only
        print(f"  [Naive Fine-tuning] Step 2: Fine-tune on {len(X_retain)} retain samples...")
        self.retrained_model = fit_distribution_to_data(
            X_retain,
            method=self.method,
            n_components=self.n_components
        )
        
        # Mix the two models (weighted combination)
        if self.method == 'gmm':
            # Combine GMM components
            all_weights = np.concatenate([
                self.original_model.weights * (1 - self.finetune_strength),
                self.retrained_model.weights * self.finetune_strength
            ])
            all_weights = all_weights / all_weights.sum()
            
            all_components = self.original_model.components + self.retrained_model.components
            
            self.model = GaussianMixtureDistribution(all_weights, all_components)
        else:
            # For KDE, just use the retrained model
            self.model = self.retrained_model
        
        print(f"  [Naive Fine-tuning] ✓ Fine-tuning complete (α={self.finetune_strength})")


class RandomModel(BaselineUnlearner):
    """
    Random Baseline.
    
    A model that generates random noise, representing the worst possible outcome.
    Useful for sanity checks.
    """
    
    def fit(self, X_retain: np.ndarray, X_forget: np.ndarray = None) -> None:
        """
        Create a random model based on data statistics.
        
        Args:
            X_retain: Retain data (used only for estimating data range)
            X_forget: Ignored
        """
        print(f"  [Random Model] Creating random baseline...")
        
        # Estimate data range
        data_mean = X_retain.mean(axis=0)
        data_std = X_retain.std(axis=0) * 3  # Wider spread
        
        # Create random components
        rng = np.random.RandomState(42)
        weights = rng.dirichlet(np.ones(self.n_components))
        
        components = []
        for _ in range(self.n_components):
            mean = data_mean + rng.randn(len(data_mean)) * data_std
            cov = np.eye(len(data_mean)) * (data_std**2).mean()
            components.append({'mean': mean, 'cov': cov})
        
        self.model = GaussianMixtureDistribution(weights, components)
        print(f"  [Random Model] ✓ Random model created")


def evaluate_baselines(X_retain: np.ndarray, X_forget: np.ndarray,
                       method: str = 'gmm', n_components: int = 5,
                       n_samples: int = 5000) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline methods.
    
    Args:
        X_retain: Retain data
        X_forget: Forget data
        method: Distribution fitting method
        n_components: Number of components
        n_samples: Number of samples for metric computation
    
    Returns:
        Dictionary mapping baseline names to their evaluation metrics
    """
    print("\n" + "=" * 80)
    print("EVALUATING BASELINE METHODS")
    print("=" * 80)
    
    # Exact unlearning (gold standard)
    exact = ExactUnlearning(method=method, n_components=n_components)
    exact.fit(X_retain, X_forget)
    P_ideal = exact.get_distribution()
    
    # Original model (no unlearning)
    original = OriginalModel(method=method, n_components=n_components)
    original.fit(X_retain, X_forget)
    
    # Naive fine-tuning
    naive = NaiveFinetuning(method=method, n_components=n_components, finetune_strength=0.7)
    naive.fit(X_retain, X_forget)
    
    # Random model
    random_model = RandomModel(method=method, n_components=n_components)
    random_model.fit(X_retain, X_forget)
    
    baselines = {
        'Exact Unlearning (Gold Standard)': exact,
        'Original Model (No Unlearning)': original,
        'Naive Fine-tuning': naive,
        'Random Baseline': random_model
    }
    
    print("\n" + "=" * 80)
    print("COMPUTING BASELINE METRICS")
    print("=" * 80)
    
    results = {}
    
    for name, baseline in baselines.items():
        print(f"\n[{name}]")
        Q = baseline.get_distribution()
        
        # Compute metrics
        metrics = {}
        
        # Core PAC metrics
        if name == 'Exact Unlearning (Gold Standard)':
            # Exact unlearning compared to itself should be 0
            metrics['epsilon'] = 0.0
            metrics['lambda'] = 0.0
        else:
            metrics['epsilon'] = total_variation_distance(Q, P_ideal, n_samples=n_samples)
            metrics['lambda'] = unlearning_condition(Q, X_forget, n_samples=n_samples)
        
        metrics['satisfies_pac'] = metrics['epsilon'] <= 0.5 and metrics['lambda'] <= metrics['epsilon']
        
        # Additional metrics
        if name != 'Exact Unlearning (Gold Standard)':
            metrics['kl_divergence'] = kl_divergence(P_ideal, Q, n_samples=n_samples)
            metrics['js_divergence'] = js_divergence(P_ideal, Q, n_samples=n_samples)
            metrics['hellinger'] = hellinger_distance(P_ideal, Q, n_samples=n_samples)
            metrics['wasserstein'] = wasserstein_distance(P_ideal, Q, n_samples=min(5000, n_samples))
        else:
            metrics['kl_divergence'] = 0.0
            metrics['js_divergence'] = 0.0
            metrics['hellinger'] = 0.0
            metrics['wasserstein'] = 0.0
        
        results[name] = metrics
        
        # Print results
        print(f"  ε (TV):        {metrics['epsilon']:.4f}")
        print(f"  λ (forget):    {metrics['lambda']:.4f}")
        print(f"  PAC satisfied: {'✓' if metrics['satisfies_pac'] else '✗'}")
        print(f"  KL divergence: {metrics['kl_divergence']:.4f}")
        print(f"  JS divergence: {metrics['js_divergence']:.4f}")
    
    return results, P_ideal


def compare_with_baselines(bayesian_metrics: Dict[str, float],
                           baseline_results: Dict[str, Dict[str, float]]) -> None:
    """
    Print a comparison table of Bayesian unlearning vs baselines.
    
    Args:
        bayesian_metrics: Metrics from Bayesian unlearning
        baseline_results: Results from baseline evaluation
    """
    print("\n" + "=" * 100)
    print("COMPARISON: BAYESIAN UNLEARNING vs BASELINES")
    print("=" * 100)
    
    # Print header
    header = f"{'Method':<35} {'ε (TV)':<10} {'λ (forget)':<12} {'KL-Div':<10} {'JS-Div':<10} {'PAC?':<8}"
    print(header)
    print("-" * 100)
    
    # Print baselines
    for name, metrics in baseline_results.items():
        pac_status = "✓" if metrics['satisfies_pac'] else "✗"
        print(f"{name:<35} {metrics['epsilon']:<10.4f} {metrics['lambda']:<12.4f} "
              f"{metrics['kl_divergence']:<10.4f} {metrics['js_divergence']:<10.4f} {pac_status:<8}")
    
    # Print Bayesian unlearning
    pac_status = "✓" if bayesian_metrics['satisfies_pac'] else "✗"
    print(f"{'Bayesian Unlearning (Ours)':<35} {bayesian_metrics['epsilon']:<10.4f} "
          f"{bayesian_metrics['lambda']:<12.4f} {bayesian_metrics.get('kl_divergence', 0):<10.4f} "
          f"{bayesian_metrics.get('js_divergence', 0):<10.4f} {pac_status:<8}")
    
    print("=" * 100)
    
    # Analysis
    print("\n📊 ANALYSIS:")
    print("-" * 100)
    
    # Compare with exact unlearning
    exact_eps = baseline_results['Exact Unlearning (Gold Standard)']['epsilon']
    bayesian_eps = bayesian_metrics['epsilon']
    print(f"Distance from Gold Standard (Exact Unlearning):")
    print(f"  Bayesian: ε = {bayesian_eps:.4f}")
    
    # Compare with no unlearning
    no_unlearn_lambda = baseline_results['Original Model (No Unlearning)']['lambda']
    bayesian_lambda = bayesian_metrics['lambda']
    print(f"\nForgetting Effectiveness (vs No Unlearning):")
    print(f"  Original Model: λ = {no_unlearn_lambda:.4f} (worst case)")
    print(f"  Bayesian: λ = {bayesian_lambda:.4f}")
    print(f"  Improvement: {((no_unlearn_lambda - bayesian_lambda) / no_unlearn_lambda * 100):.1f}% reduction in forget mass")
    
    # Compare with naive fine-tuning
    naive_eps = baseline_results['Naive Fine-tuning']['epsilon']
    print(f"\nUtility Comparison (vs Naive Fine-tuning):")
    print(f"  Naive: ε = {naive_eps:.4f}")
    print(f"  Bayesian: ε = {bayesian_eps:.4f}")
    if bayesian_eps < naive_eps:
        print(f"  ✓ Bayesian is {((naive_eps - bayesian_eps) / naive_eps * 100):.1f}% better")
    else:
        print(f"  Naive is {((bayesian_eps - naive_eps) / naive_eps * 100):.1f}% better")
    
    print("=" * 100)


if __name__ == "__main__":
    print("Baseline Unlearning Methods")
    print("=" * 60)
    print("Available baselines:")
    print("  1. ExactUnlearning - Gold standard (retrain from scratch)")
    print("  2. OriginalModel - No unlearning (worst case)")
    print("  3. NaiveFinetuning - Simple fine-tuning approach")
    print("  4. RandomModel - Random baseline (sanity check)")
    print("\nUse evaluate_baselines() to compare all methods.")
