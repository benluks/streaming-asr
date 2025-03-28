import pickle
from matplotlib import cm, pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns


ENCDOING_FILE = "vowel_encodings/vowel_encodings.pkl"


def load_encoding(encoding_path):

    with open(encoding_path, "rb") as f:
        data = pickle.load(f)

    X = data["encodings"].T
    labels = data["labels"]
    lengths = data["lengths"]

    return X, labels, lengths


def run_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    return pca, cumulative


def plot_pairwise(X_pca, labels, lengths):

    df = pd.DataFrame(X_pca[:, :10], columns=[f"PC{i+1}" for i in range(10)])

    label_ids = np.repeat(np.arange(len(lengths)), lengths)
    df["vowel"] = np.array(labels)[label_ids]

    sns.pairplot(df, hue="vowel", corner=True, plot_kws={"s": 10, "alpha": 0.6})
    plt.suptitle("Pairwise PCA Component Plots (First 10 PCs)", y=1.02)
    plt.show()


def plot_pca(X_pca, labels, lengths):
    label_ids = np.repeat(np.arange(len(lengths)), lengths)

    plt.figure(figsize=(12, 8))
    sc = plt.scatter(X_pca[:, 1], X_pca[:, 5], c=label_ids, cmap="tab20", s=10)

    cbar = plt.colorbar(sc, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("Vowel")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Encoder Output (Colored by Vowel)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca(cumulative):
    """
    Plot the PCA of the full dataset."
    """

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o")
    plt.axhline(y=0.9, color="red", linestyle="--", label="90% variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    X, labels, lengths = load_encoding(
        "vowel_encodings/vowel_encodings_common_relu.pkl"
    )

    pca, cumulative = run_pca(X)
    X_pca = pca.transform(X)

    plot_pairwise(X_pca, labels, lengths)
