"""
Quick test script to verify the framework is working correctly.
This is a minimal test that doesn't require visualization.
"""

import numpy as np
from src.synthetic import generate_moon_dataset, TargetConcept, partition_data
from src.bayesian import BayesianUnlearning, GaussianPrior, fit_distribution_to_data


def test_framework():
    print("=" * 60)
    print("QUICK TEST: Bayesian PAC Unlearning Framework")
    print("=" * 60)
    
    # 1. Generate data
    print("\n[1/5] Generating synthetic data...")
    X, y = generate_moon_dataset(n_samples=300, noise=0.1, random_state=42)
    concept = TargetConcept.spatial_region(x_min=0.5, x_max=2.0, y_min=-0.3, y_max=0.7)
    X_retain, X_forget, partition = partition_data(X, y, concept)
    print(f"  ✓ Generated {len(X)} samples")
    print(f"    - Retain: {len(X_retain)} samples")
    print(f"    - Forget: {len(X_forget)} samples")
    
    # 2. Fit ideal distribution
    print("\n[2/5] Fitting ideal retain distribution...")
    P_ideal = fit_distribution_to_data(X_retain, method='gmm', n_components=3)
    print(f"  ✓ Fitted {P_ideal.n_components}-component GMM")
    
    # 3. Setup prior
    print("\n[3/5] Setting up prior...")
    data_mean = X_retain.mean(axis=0)
    data_cov = np.cov(X_retain.T) * 2
    prior = GaussianPrior(mean_prior_mean=data_mean, 
                         mean_prior_cov=data_cov,
                         dim=2, n_components=3)
    print(f"  ✓ GaussianPrior initialized")
    
    # 4. Bayesian unlearning
    print("\n[4/5] Running Bayesian unlearning...")
    unlearner = BayesianUnlearning(prior=prior, lambda_hyper=2.0)
    unlearner.fit(X_retain, X_forget, n_posterior_samples=20, n_components=3)
    Q_unlearned = unlearner.get_posterior_predictive()
    print(f"  ✓ Posterior predictive computed ({Q_unlearned.n_components} components)")
    
    # 5. Evaluate
    print("\n[5/5] Evaluating PAC metrics...")
    metrics = unlearner.evaluate(P_ideal, X_forget, n_samples=2000)
    print(f"  ✓ Metrics computed")
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"  ε (TV distance):        {metrics['epsilon']:.4f}")
    print(f"  λ (forget mass):        {metrics['lambda']:.4f}")
    print(f"  ε-MC satisfied (ε≤0.5): {metrics['epsilon'] <= 0.5}")
    print(f"  λ-UC satisfied (λ≤ε):   {metrics['lambda'] <= metrics['epsilon']}")
    print(f"  PAC condition met:      {metrics['satisfies_pac']}")
    print("=" * 60)
    
    # Verify basic properties
    assert 0 <= metrics['epsilon'] <= 1, "ε should be in [0,1]"
    assert 0 <= metrics['lambda'] <= 1, "λ should be in [0,1]"
    
    print("\n✅ All tests passed!")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = test_framework()
        print("\n✅ SUCCESS: Framework is working correctly!")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
