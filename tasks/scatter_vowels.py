import argparse
import numpy as np

from utils.io import load_encodings_from_pkl, load_pickle
from utils.stats import run_pca, run_tsne
from utils.plotting import plot_scatter


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scatter plot specific vowels")
    parser.add_argument(
        "--vowels",
        "-v",
        nargs="+",
        required=True,
        help="List of vowels to include (e.g., --vowels i u æ)",
    )
    parser.add_argument(
        "--pkl",
        "-p",
        type=str,
        required=True,
        help="Path to the encoding_dict pickle file",
    )
    parser.add_argument("--tsne", action="store_true", help="Apply t-SNE on top of PCA")
    parser.add_argument(
        "--relu", action="store_true", help="Apply ReLU to encodings before PCA"
    )
    parser.add_argument(
        "--pca_source",
        type=str,
        default="vowel_encodings/vowel_encodings_common.pkl",
        help="Pickled encodings file to run PCA on",
    )

    args = parser.parse_args()
    

    X, *_ = load_encodings_from_pkl(args.pca_source)
    encoding_dict = load_pickle(args.pkl)

    vowel_encodings, lengths = zip(
        *[(enc, enc.shape[0]) for v in args.vowels if (enc := encoding_dict[v]).all()]
    )

    encodings_concat = np.concatenate(vowel_encodings, axis=0)

    pca, cumulative = run_pca(X)
    X_transformed = pca.transform(encodings_concat)

    reduction = "PCA"
    if args.tsne:
        X_transformed = run_tsne(X_transformed, n_components=2, starting_dims=50)
        reduction = "t-SNE"
    if args.relu:
        encodings_full = np.maximum(X_transformed, 0)

    plot_scatter(X_transformed, args.vowels, lengths)
