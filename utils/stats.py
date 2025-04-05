import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def run_pca(X, n_components=None):
    pca = PCA(n_components=n_components)
    pca.fit(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    return pca, cumulative


def run_tsne(X_pca, n_components=2, starting_dims=50):
    tsne = TSNE(
        n_components=n_components,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        # random_state=42,
    )
    X_tsne = tsne.fit_transform(X_pca[:, :starting_dims])
    return X_tsne


def get_label_ids(lengths):
    """
    Get label ids for the lengths of each vowel.
    """
    # Create a list of label ids for each vowel
    return np.repeat(np.arange(len(lengths)), lengths)
