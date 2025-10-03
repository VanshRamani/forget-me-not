"""
Synthetic dataset generation for PAC-Bayesian Unlearning Framework.

This module provides modular functions to create synthetic datasets with
clearly defined forget and retain regions for testing unlearning algorithms.
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from typing import Tuple, Callable, Optional
import matplotlib.pyplot as plt


class TargetConcept:
    """
    Defines a target concept h: X -> {0, 1} that partitions the data space.
    h(x) = 1 indicates the forget region X₁
    h(x) = 0 indicates the retain region X₀
    """
    
    def __init__(self, concept_fn: Callable[[np.ndarray], np.ndarray], name: str = "custom"):
        """
        Args:
            concept_fn: Function that takes array of shape (n, d) and returns array of shape (n,)
                       with 0s (retain) and 1s (forget)
            name: Name of the concept for visualization
        """
        self.concept_fn = concept_fn
        self.name = name
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Apply the concept to partition data."""
        return self.concept_fn(X)
    
    @staticmethod
    def spatial_region(x_min: float, x_max: float, y_min: float, y_max: float) -> 'TargetConcept':
        """Forget region defined by a spatial rectangular region."""
        def concept_fn(X):
            in_region = (X[:, 0] >= x_min) & (X[:, 0] <= x_max) & \
                       (X[:, 1] >= y_min) & (X[:, 1] <= y_max)
            return in_region.astype(int)
        return TargetConcept(concept_fn, name="spatial_region")
    
    @staticmethod
    def radial(center: Tuple[float, float], radius: float) -> 'TargetConcept':
        """Forget region defined by a circular region."""
        def concept_fn(X):
            distances = np.sqrt((X[:, 0] - center[0])**2 + (X[:, 1] - center[1])**2)
            return (distances <= radius).astype(int)
        return TargetConcept(concept_fn, name="radial")
    
    @staticmethod
    def halfspace(normal: np.ndarray, offset: float = 0.0) -> 'TargetConcept':
        """Forget region defined by a halfspace (linear boundary)."""
        def concept_fn(X):
            return (X @ normal + offset > 0).astype(int)
        return TargetConcept(concept_fn, name="halfspace")
    
    @staticmethod
    def class_based(class_idx: int) -> 'TargetConcept':
        """Forget region defined by specific class labels (for classification datasets)."""
        def concept_fn(X, y=None):
            if y is None:
                raise ValueError("class_based concept requires labels")
            return (y == class_idx).astype(int)
        return TargetConcept(concept_fn, name=f"class_{class_idx}")


def generate_moon_dataset(n_samples: int = 1000, noise: float = 0.1, 
                          random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two interleaving half circles (moons).
    
    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        random_state: Random seed for reproducibility
    
    Returns:
        X: Data points of shape (n_samples, 2)
        y: Class labels of shape (n_samples,)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def generate_circles_dataset(n_samples: int = 1000, noise: float = 0.05,
                            factor: float = 0.5, random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circles.
    
    Args:
        n_samples: Total number of samples
        noise: Standard deviation of Gaussian noise
        factor: Scale factor between inner and outer circle
        random_state: Random seed for reproducibility
    
    Returns:
        X: Data points of shape (n_samples, 2)
        y: Class labels of shape (n_samples,)
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y


def generate_blobs_dataset(n_samples: int = 1000, n_centers: int = 3,
                          cluster_std: float = 0.5, center_box: Tuple[float, float] = (-3, 3),
                          random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate isotropic Gaussian blobs.
    
    Args:
        n_samples: Total number of samples
        n_centers: Number of cluster centers
        cluster_std: Standard deviation of clusters
        center_box: Bounding box for cluster centers
        random_state: Random seed for reproducibility
    
    Returns:
        X: Data points of shape (n_samples, 2)
        y: Cluster labels of shape (n_samples,)
    """
    X, y = make_blobs(n_samples=n_samples, n_features=2, centers=n_centers,
                     cluster_std=cluster_std, center_box=center_box, random_state=random_state)
    return X, y


def generate_gaussian_mixture(n_samples: int = 1000, 
                             centers: Optional[np.ndarray] = None,
                             covs: Optional[list] = None,
                             weights: Optional[np.ndarray] = None,
                             random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate samples from a custom Gaussian mixture model.
    
    Args:
        n_samples: Total number of samples
        centers: Array of shape (n_components, n_features) with component means
        covs: List of covariance matrices, one per component
        weights: Mixture weights (must sum to 1)
        random_state: Random seed for reproducibility
    
    Returns:
        X: Data points of shape (n_samples, 2)
        y: Component labels of shape (n_samples,)
    """
    rng = np.random.RandomState(random_state)
    
    # Default: 2 Gaussian components
    if centers is None:
        centers = np.array([[-2, -2], [2, 2]])
    if covs is None:
        covs = [np.eye(2) * 0.5, np.eye(2) * 0.5]
    if weights is None:
        weights = np.ones(len(centers)) / len(centers)
    
    # Sample component assignments
    component_assignments = rng.choice(len(centers), size=n_samples, p=weights)
    
    # Generate samples from each component
    X = np.zeros((n_samples, centers.shape[1]))
    for i in range(n_samples):
        comp = component_assignments[i]
        X[i] = rng.multivariate_normal(centers[comp], covs[comp])
    
    return X, component_assignments


def partition_data(X: np.ndarray, y: np.ndarray, 
                   concept: TargetConcept) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Partition dataset into retain and forget regions based on target concept.
    
    Args:
        X: Data points of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
        concept: TargetConcept that defines the partition
    
    Returns:
        X_retain (D₀): Points in retain region (h(x) = 0)
        X_forget (D₁): Points in forget region (h(x) = 1)
        partition_labels: Array indicating region (0 or 1) for each point
    """
    partition_labels = concept(X)
    
    X_retain = X[partition_labels == 0]
    X_forget = X[partition_labels == 1]
    
    return X_retain, X_forget, partition_labels


def visualize_dataset(X: np.ndarray, partition_labels: np.ndarray, 
                     title: str = "Dataset with Forget/Retain Regions",
                     save_path: Optional[str] = None):
    """
    Visualize the dataset with forget and retain regions.
    
    Args:
        X: Data points of shape (n_samples, 2)
        partition_labels: Array of 0s (retain) and 1s (forget)
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Plot retain region (blue)
    retain_mask = partition_labels == 0
    plt.scatter(X[retain_mask, 0], X[retain_mask, 1], 
               c='blue', alpha=0.6, label='Retain Region (X₀)', s=30)
    
    # Plot forget region (red)
    forget_mask = partition_labels == 1
    plt.scatter(X[forget_mask, 0], X[forget_mask, 1], 
               c='red', alpha=0.6, label='Forget Region (X₁)', s=30)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Example 1: Moons dataset with spatial region concept
    print("=" * 60)
    print("Example 1: Moons Dataset with Spatial Region")
    print("=" * 60)
    X, y = generate_moon_dataset(n_samples=500, noise=0.1)
    concept = TargetConcept.spatial_region(x_min=-0.5, x_max=1.5, y_min=-0.5, y_max=0.5)
    X_retain, X_forget, partition = partition_data(X, y, concept)
    print(f"Total samples: {len(X)}")
    print(f"Retain samples: {len(X_retain)}")
    print(f"Forget samples: {len(X_forget)}")
    visualize_dataset(X, partition, "Moons Dataset with Spatial Forget Region")
    
    # Example 2: Circles dataset with radial concept
    print("\n" + "=" * 60)
    print("Example 2: Circles Dataset with Radial Concept")
    print("=" * 60)
    X, y = generate_circles_dataset(n_samples=500, noise=0.05)
    concept = TargetConcept.radial(center=(0, 0), radius=0.5)
    X_retain, X_forget, partition = partition_data(X, y, concept)
    print(f"Total samples: {len(X)}")
    print(f"Retain samples: {len(X_retain)}")
    print(f"Forget samples: {len(X_forget)}")
    visualize_dataset(X, partition, "Circles Dataset with Radial Forget Region")
    
    # Example 3: Blobs dataset with halfspace concept
    print("\n" + "=" * 60)
    print("Example 3: Blobs Dataset with Halfspace Concept")
    print("=" * 60)
    X, y = generate_blobs_dataset(n_samples=600, n_centers=4)
    concept = TargetConcept.halfspace(normal=np.array([1, 1]), offset=0)
    X_retain, X_forget, partition = partition_data(X, y, concept)
    print(f"Total samples: {len(X)}")
    print(f"Retain samples: {len(X_retain)}")
    print(f"Forget samples: {len(X_forget)}")
    visualize_dataset(X, partition, "Blobs Dataset with Halfspace Forget Region")