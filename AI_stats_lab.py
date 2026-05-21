"""
AI_stats_lab.py

Lab: Unsupervised Learning and K-Means Clustering
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans


# ============================================================
# Question 1: Unlabeled Data and K-Means Clustering
# ============================================================

def load_iris_unlabeled(feature_indices=(0, 1)):
    iris = load_iris()
    X = iris.data[:, list(feature_indices)]
    feature_names = [iris.feature_names[i] for i in feature_indices]
    return {"X": X, "feature_names": feature_names}


def standardize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std_safe
    return {"X_scaled": X_scaled, "mean": mean, "std": std_safe}


def fit_kmeans(X, K, random_state=0, n_init=10):
    model = KMeans(n_clusters=K, random_state=random_state, n_init=n_init)
    model.fit(X)
    return {
        "centroids": model.cluster_centers_,
        "labels": model.labels_,
        "objective": model.inertia_,
        "n_iter": model.n_iter_
    }


def compute_kmeans_objective(X, centroids, labels):
    diffs = X - centroids[labels]
    return float(np.sum(np.sum(diffs ** 2, axis=1)))


# ============================================================
# Question 2: Choosing K, Underfitting/Overfitting, and Outliers
# ============================================================

def evaluate_k_values(X, k_values, random_state=0, n_init=10):
    objectives = []
    for k in k_values:
        result = fit_kmeans(X, K=k, random_state=random_state, n_init=n_init)
        objectives.append(result["objective"])

    relative_improvements = [0.0]
    for i in range(1, len(objectives)):
        prev = objectives[i - 1]
        curr = objectives[i]
        improvement = (prev - curr) / prev if prev != 0 else 0.0
        relative_improvements.append(improvement)

    return {
        "k_values": k_values,
        "objectives": objectives,
        "relative_improvements": relative_improvements
    }


def choose_elbow_k(k_values, objectives):
    if len(k_values) < 3:
        return k_values[0]

    # Endpoints of the line
    x0, y0 = 0, objectives[0]
    xn, yn = len(k_values) - 1, objectives[-1]

    # Line vector
    dx = xn - x0
    dy = yn - y0
    line_len = np.sqrt(dx ** 2 + dy ** 2)

    max_dist = -1
    best_idx = 1

    for i in range(1, len(k_values) - 1):
        # Perpendicular distance from point (i, objectives[i]) to the line
        px = i - x0
        py = objectives[i] - y0
        dist = abs(px * dy - py * dx) / line_len
        if dist > max_dist:
            max_dist = dist
            best_idx = i

    return k_values[best_idx]


def cluster_size_summary(labels, K):
    return {k: int(np.sum(labels == k)) for k in range(K)}


def identify_outliers_by_distance(X, centroids, labels, top_n=5):
    diffs = X - centroids[labels]
    distances = np.sum(diffs ** 2, axis=1)
    sorted_indices = np.argsort(distances)[::-1]
    top_indices = sorted_indices[:top_n]
    return {
        "indices": top_indices,
        "distances": distances[top_indices]
    }


def diagnose_clustering_fit(K, elbow_k):
    if K < elbow_k:
        return "underfitting"
    elif K == elbow_k:
        return "good_fit"
    else:
        return "overfitting"


# ============================================================
# Question 3: Visualization
# ============================================================

def plot_unlabeled_data(X, feature_names=None, title="Unlabeled Data"):
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], alpha=0.6)
    ax.set_title(title)
    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    return fig, ax


def plot_kmeans_clusters(X, labels, centroids, feature_names=None, title="K-Means Clusters"):
    fig, ax = plt.subplots()
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], alpha=0.6, label=f"Cluster {label}")
    ax.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200, c="black", zorder=5, label="Centroids")
    ax.set_title(title)
    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
    ax.legend()
    return fig, ax


def plot_elbow_curve(k_values, objectives, title="Elbow Method"):
    fig, ax = plt.subplots()
    ax.plot(k_values, objectives, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Number of clusters K")
    ax.set_ylabel("Objective value")
    return fig, ax


if __name__ == "__main__":
    print("Implement all required functions.")
