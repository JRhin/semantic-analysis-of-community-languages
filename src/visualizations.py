"""
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ========================================================================
#
#                        FUNCTIONS DEFINITION
#
# ========================================================================

def plot_explained_variance(pca: PCA) -> None:
    """
    Plot the variance explained by each PCA component.

    Args:
      pca : PCA
        PCA object, fitted PCA model
      
    Returns:
      None
    """
    explained_variance = pca.explained_variance_ratio_ * 100  # Convert to percentage
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center',
            label='Individual Explained Variance')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid',
             label='Cumulative Explained Variance', color='red')
    plt.title('Explained Variance by PCA Components')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Explained Variance (%)')
    plt.legend(loc='best')
    plt.show()
    return None


def plot_elbow_and_silhouette(elbow_metrics: list[float],
                              silhouette_metrics: list[np.float32],
                              max_clusters: int) -> None:
    """
    Plot the elbow and silhouette scores.

    Args:
      elbow_metrics : list[float]
        inertia values for the elbow method
      silhouette_metrics: list[numpy.float32]
        silhouette scores
      max_clusters : int
        the maximum number of clusters to consider

    Returns:
      None
    """
    clusters_range = range(2, max_clusters + 1)

    # Plot the elbow method
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(clusters_range, elbow_metrics, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

    # Plot the silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(clusters_range, silhouette_metrics, marker='o')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')

    plt.tight_layout()
    plt.show()

    return None


def plot_clusters(pca_data: np.ndarray,
                  kmeans_labels: np.ndarray,
                  n_clusters: int) -> None:
    """
    Plot clusters based on the first two PCA dimensions.

    Args:
      pca_data : numpy.ndarray
        data after PCA transformation
      kmeans_labels : np.ndarray
        cluster labels from K-means
      n_clusters : int
        the number of clusters

    Returns:
      None
    """
    plt.figure(figsize=(8, 6))

    # Plot each cluster with a unique color
    for cluster in range(n_clusters):
        cluster_points = pca_data[kmeans_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}')

    # Add cluster centers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pca_data)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, label='Centroids')

    plt.title('Clusters in the First Two PCA Dimensions')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend()
    plt.show()

    return None



# ========================================================================
#
#                                MAIN LOOP
#
# ========================================================================

def main() -> None:
    """The main loop.
    """

    return None


if __name__ == "__main__":
    main()
