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


def plot_pca_pairwise(X_pca, labels, lengths, pc_count=10):
    label_ids = np.repeat(np.arange(len(lengths)), lengths)

    total_axes = pc_count * (pc_count - 1) // 2
    rows = pc_count - 1
    cols = pc_count - 1

    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), sharex="col", sharey="row")
    fig.suptitle("Pairwise PCA Component Projections", fontsize=16, y=0.92)
    for i in range(pc_count):
        for j in range(i + 1, pc_count):
            try:
                ax = axes[i, j - 1]
            except IndexError as e:
                print(i, j)
                raise e

            ax.scatter(
                X_pca[:, i], X_pca[:, j], c=label_ids, cmap="tab20", s=8, alpha=0.6
            )

    # add colorbar using a dummy as mappable
    dummy = axes[0, 0].scatter(
        [], [], c=[], cmap="tab20"
    )  # empty but with same colormap
    cbar = plt.colorbar(dummy, ax=axes, ticks=np.arange(len(labels)))
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("Vowel")

    # Add global axis labels to the bottom row and first column
    for j in range(rows):
        axes[-1, j - 1].set_xlabel(f"PC{j+1}", fontsize=9)
    for i in range(cols):
        axes[i, 0].set_ylabel(f"PC{i+1}", fontsize=9)

    plt.title("PCA of Encoder Output (Colored by Vowel)")
    # plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_component_tracks(X_pca, labels, lengths, n_components=10, num_vowels=None):
    """
    Plot the PCA of the full dataset with vertical lines for components.
    Additionally, add labels for each vowel.
    """
    if num_vowels is None:
        num_vowels = len(labels)
    boundaries = np.cumsum([0] + lengths[:num_vowels])

    plt.figure(figsize=(20, 8))
    plt.plot(
        X_pca[: boundaries[-1], :n_components],
        marker="o",
        markersize=2,
        alpha=0.5,
        label="PCA Components",
    )
    plt.xlabel("Time step")
    plt.ylabel("PCA Component")

    # Draw vertical lines and labels
    for i, b in enumerate(boundaries[:-1]):
        plt.axvline(x=b, color="red", linestyle="--", linewidth=1)

    midpoints = [
        (boundaries[i] + boundaries[i + 1]) // 2 for i in range(len(boundaries) - 1)
    ]
    for mid, label in zip(midpoints, labels):
        plt.text(mid, -5, label, ha="center", va="bottom", fontsize=9)

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

    plot_component_tracks(X_pca, labels, lengths, n_components=10, num_vowels=1)
