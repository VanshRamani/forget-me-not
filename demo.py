"""
Demo script for Bayesian PAC Unlearning Framework.

This script demonstrates the complete workflow:
1. Generate synthetic data with forget/retain regions
2. Fit ideal retain distribution
3. Apply Bayesian unlearning
4. Evaluate using PAC metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from src.synthetic import (
    generate_moon_dataset, generate_blobs_dataset,
    TargetConcept, partition_data, visualize_dataset
)
from src.bayesian import (
    BayesianUnlearning, GaussianPrior, DirichletProcessPrior,
    fit_distribution_to_data, GaussianMixtureDistribution,
    total_variation_distance
)

# Ensure output directory exists
os.makedirs('./figs', exist_ok=True)


def visualize_distributions(X_retain, X_forget, Q_unlearned, P_ideal, 
                           title="Unlearning Results", save_path="./figs/"):
    """Visualize the unlearning results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Create grid for density visualization
    x_min, x_max = X_retain[:, 0].min() - 1, X_retain[:, 0].max() + 1
    y_min, y_max = X_retain[:, 1].min() - 1, X_retain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Plot 1: Data partitioning
    axes[0].scatter(X_retain[:, 0], X_retain[:, 1], c='blue', alpha=0.6, 
                   label='Retain (D₀)', s=30)
    axes[0].scatter(X_forget[:, 0], X_forget[:, 1], c='red', alpha=0.6, 
                   label='Forget (D₁)', s=30)
    axes[0].set_title('Data Partitioning')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Plot 2: Ideal retain distribution
    Z_ideal = P_ideal.pdf(grid).reshape(xx.shape)
    axes[1].contourf(xx, yy, Z_ideal, levels=15, cmap='Blues', alpha=0.6)
    axes[1].scatter(X_retain[:, 0], X_retain[:, 1], c='blue', alpha=0.3, s=20)
    axes[1].set_title('Ideal Retain Distribution P₀')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    
    # Plot 3: Unlearned distribution
    Z_unlearned = Q_unlearned.pdf(grid).reshape(xx.shape)
    axes[2].contourf(xx, yy, Z_unlearned, levels=15, cmap='Greens', alpha=0.6)
    axes[2].scatter(X_retain[:, 0], X_retain[:, 1], c='blue', alpha=0.3, s=20)
    axes[2].scatter(X_forget[:, 0], X_forget[:, 1], c='red', alpha=0.3, s=20, marker='x')
    axes[2].set_title('Unlearned Distribution Q')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        path = save_path + title + ".png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def run_unlearning_experiment(dataset_name='moons', concept_type='spatial',
                              lambda_hyper=1.0, n_posterior_samples=50):
    """
    Run a complete unlearning experiment.
    
    Args:
        dataset_name: 'moons', 'blobs', or 'circles'
        concept_type: 'spatial', 'radial', or 'halfspace'
        lambda_hyper: Forgetting strength
        n_posterior_samples: Number of posterior samples
    """
    print("=" * 80)
    print(f"BAYESIAN PAC UNLEARNING EXPERIMENT")
    print(f"Dataset: {dataset_name} | Concept: {concept_type} | λ_hyper: {lambda_hyper}")
    print("=" * 80)
    
    # Step 1: Generate dataset
    print("\n[Step 1] Generating synthetic dataset...")
    if dataset_name == 'moons':
        X, y = generate_moon_dataset(n_samples=800, noise=0.1)
        if concept_type == 'spatial':
            concept = TargetConcept.spatial_region(x_min=0.5, x_max=2.0, y_min=-0.3, y_max=0.7)
        elif concept_type == 'radial':
            concept = TargetConcept.radial(center=(1.0, 0.0), radius=0.7)
        else:
            concept = TargetConcept.halfspace(normal=np.array([1, 0]), offset=-0.5)
    elif dataset_name == 'blobs':
        X, y = generate_blobs_dataset(n_samples=800, n_centers=4, cluster_std=0.6)
        if concept_type == 'spatial':
            concept = TargetConcept.spatial_region(x_min=0, x_max=5, y_min=-2, y_max=2)
        else:
            concept = TargetConcept.halfspace(normal=np.array([1, 1]), offset=0)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    X_retain, X_forget, partition = partition_data(X, y, concept)
    print(f"  Total samples: {len(X)}")
    print(f"  Retain samples: {len(X_retain)} ({100*len(X_retain)/len(X):.1f}%)")
    print(f"  Forget samples: {len(X_forget)} ({100*len(X_forget)/len(X):.1f}%)")
    
    # Step 2: Fit ideal retain distribution
    print("\n[Step 2] Fitting ideal retain distribution P₀...")
    P_ideal = fit_distribution_to_data(X_retain, method='gmm', n_components=5)
    print(f"  Fitted {P_ideal.n_components}-component GMM to retain data")
    
    # Step 3: Setup prior
    print("\n[Step 3] Setting up prior distribution...")
    data_mean = X_retain.mean(axis=0)
    data_cov = np.cov(X_retain.T) * 2
    prior = GaussianPrior(mean_prior_mean=data_mean, 
                         mean_prior_cov=data_cov,
                         dim=2, n_components=5)
    print(f"  Using GaussianPrior centered at {data_mean}")
    
    # Step 4: Bayesian unlearning
    print("\n[Step 4] Performing Bayesian unlearning...")
    unlearner = BayesianUnlearning(prior=prior, lambda_hyper=lambda_hyper)
    unlearner.fit(X_retain, X_forget, 
                 n_posterior_samples=n_posterior_samples,
                 n_components=5)
    
    Q_unlearned = unlearner.get_posterior_predictive()
    print(f"  Posterior predictive has {Q_unlearned.n_components} components")
    
    # Step 5: Evaluate PAC metrics and standard unlearning metrics
    print("\n[Step 5] Evaluating unlearning metrics...")
    metrics = unlearner.evaluate(P_ideal, X_forget, n_samples=5000, compute_all_metrics=True)
    
    print(f"\n{'=' * 80}")
    print("RESULTS:")
    print(f"{'=' * 80}")
    print("\n📊 PAC Framework Metrics:")
    print(f"  ε (TV distance to P₀):      {metrics['epsilon']:.4f}")
    print(f"  λ (forget region mass):      {metrics['lambda']:.4f}")
    print(f"  Satisfies ε-MC (ε ≤ 0.5):    {metrics['epsilon'] <= 0.5}")
    print(f"  Satisfies λ-UC (λ ≤ ε):      {metrics['lambda'] <= metrics['epsilon']}")
    print(f"  Overall PAC condition:       {'✓ PASS' if metrics['satisfies_pac'] else '✗ FAIL'}")
    
    print("\n📏 Additional Distance Metrics:")
    print(f"  KL Divergence D_KL(P₀||Q):   {metrics.get('kl_divergence', 'N/A'):.4f}")
    print(f"  JS Divergence (symmetric):   {metrics.get('js_divergence', 'N/A'):.4f} (max: 0.693)")
    print(f"  Hellinger Distance:          {metrics.get('hellinger', 'N/A'):.4f} (max: 1.0)")
    print(f"  Wasserstein Distance (W₁):   {metrics.get('wasserstein', 'N/A'):.4f}")
    print(f"  Chi-Squared Distance:        {metrics.get('chi_squared', 'N/A'):.4f}")
    print(f"{'=' * 80}\n")
    
    # Step 6: Visualize
    print("[Step 6] Visualizing results...")
    visualize_distributions(X_retain, X_forget, Q_unlearned, P_ideal,
                          title=f"Unlearning: {dataset_name} | λ_hyper={lambda_hyper}")
    
    return {
        'X_retain': X_retain,
        'X_forget': X_forget,
        'P_ideal': P_ideal,
        'Q_unlearned': Q_unlearned,
        'metrics': metrics,
        'unlearner': unlearner
    }


def compare_lambda_hyper_values():
    """Compare different forgetting strengths."""
    print("\n" + "=" * 80)
    print("COMPARING DIFFERENT FORGETTING STRENGTHS (λ_hyper)")
    print("=" * 80 + "\n")
    
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    results = []
    
    for lambda_hyper in lambda_values:
        print(f"\n{'*' * 80}")
        print(f"Testing λ_hyper = {lambda_hyper}")
        print(f"{'*' * 80}")
        result = run_unlearning_experiment(dataset_name='moons', 
                                          concept_type='spatial',
                                          lambda_hyper=lambda_hyper,
                                          n_posterior_samples=40)
        results.append(result)
    
    # Extract metrics
    epsilons = [r['metrics']['epsilon'] for r in results]
    lambdas = [r['metrics']['lambda'] for r in results]
    kl_divs = [r['metrics'].get('kl_divergence', 0) for r in results]
    js_divs = [r['metrics'].get('js_divergence', 0) for r in results]
    hellinger_dists = [r['metrics'].get('hellinger', 0) for r in results]
    wasserstein_dists = [r['metrics'].get('wasserstein', 0) for r in results]
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: PAC ε-MC
    axes[0, 0].plot(lambda_values, epsilons, 'o-', linewidth=2, markersize=8, label='ε (TV distance)', color='blue')
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='ε threshold (0.5)', linewidth=1.5)
    axes[0, 0].set_xlabel('λ_hyper (Forgetting Strength)', fontsize=11)
    axes[0, 0].set_ylabel('ε (Matching Condition)', fontsize=11)
    axes[0, 0].set_title('PAC ε-MC: Utility vs Forgetting Strength', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: PAC λ-UC
    axes[0, 1].plot(lambda_values, lambdas, 's-', linewidth=2, markersize=8, 
                    color='orange', label='λ (forget mass)')
    axes[0, 1].plot(lambda_values, epsilons, 'o--', linewidth=2, markersize=8, 
                    alpha=0.5, label='ε (upper bound)', color='blue')
    axes[0, 1].set_xlabel('λ_hyper (Forgetting Strength)', fontsize=11)
    axes[0, 1].set_ylabel('λ (Unlearning Condition)', fontsize=11)
    axes[0, 1].set_title('PAC λ-UC: Forgetting Quality vs Strength', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Divergence Metrics
    axes[1, 0].plot(lambda_values, kl_divs, '^-', linewidth=2, markersize=8, 
                    label='KL Divergence', color='green')
    axes[1, 0].plot(lambda_values, js_divs, 'v-', linewidth=2, markersize=8, 
                    label='JS Divergence', color='purple')
    axes[1, 0].set_xlabel('λ_hyper (Forgetting Strength)', fontsize=11)
    axes[1, 0].set_ylabel('Divergence (nats)', fontsize=11)
    axes[1, 0].set_title('Information-Theoretic Metrics', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distance Metrics
    axes[1, 1].plot(lambda_values, hellinger_dists, 'd-', linewidth=2, markersize=8, 
                    label='Hellinger', color='red')
    axes[1, 1].plot(lambda_values, wasserstein_dists, 'p-', linewidth=2, markersize=8, 
                    label='Wasserstein-1', color='brown')
    axes[1, 1].set_xlabel('λ_hyper (Forgetting Strength)', fontsize=11)
    axes[1, 1].set_ylabel('Distance', fontsize=11)
    axes[1, 1].set_title('Geometric Distance Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figs/lambda_comparison_all_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: PAC METRICS")
    print("=" * 80)
    print(f"{'λ_hyper':<10} {'ε (TV)':<10} {'λ (forget)':<12} {'KL-Div':<10} {'JS-Div':<10} {'λ≤ε?':<8}")
    print("-" * 80)
    for lh, eps, lam, kl, js in zip(lambda_values, epsilons, lambdas, kl_divs, js_divs):
        satisfies = "✓" if lam <= eps else "✗"
        print(f"{lh:<10.1f} {eps:<10.4f} {lam:<12.4f} {kl:<10.4f} {js:<10.4f} {satisfies:<8}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run single experiment
    print("\n" + "🎯" * 40)
    print("BAYESIAN PAC UNLEARNING FRAMEWORK - DEMONSTRATION")
    print("🎯" * 40 + "\n")
    
    # Experiment 1: Basic unlearning
    result1 = run_unlearning_experiment(
        dataset_name='moons',
        concept_type='spatial',
        lambda_hyper=2.0,
        n_posterior_samples=50
    )
    
    # Experiment 2: Compare different forgetting strengths
    print("\n\n")
    compare_lambda_hyper_values()
    
    print("\n" + "✅" * 40)
    print("DEMONSTRATION COMPLETE!")
    print("✅" * 40 + "\n")
