from pathlib import Path
import pickle
import numpy as np
import torch
from torch.nn.functional import relu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

KEEP_VOWELS = [
    "i",
    "y",
    "e",
    "ø",
    "ɛ",
    "œ",
    "æ",
    "a",
    "ɑ",
    "ɒ",
    "ɔ",
    "o",
    "u",
    "ʊ",
    "ɯ",
    "ɤ",
    "ə",
    "ɜ",
    "ɪ",
    "ʌ",
]


def load_encoding(encoding_path, use_relu=False):
    encoding = torch.load(f"{encoding_path}").squeeze(0)
    if use_relu:
        encoding = relu(encoding)
    return encoding.T.numpy()


def plot_encoding(encoding, lengths=None, labels=None):
    """
    Plot the encoding of a batch of audio chunks.
    Args:
        encoding_path (str): Path to the encoding file.
    """

    plt.figure(figsize=(20, 8))
    plt.imshow(encoding, aspect="auto", cmap="magma", interpolation="nearest")
    plt.colorbar(label="Activation")
    plt.xlabel("Time step")
    plt.ylabel("Feature dimension")
    plt.title("Conformer Encoder Output", pad=20)
    plt.tight_layout()

    if lengths:
        boundaries = np.cumsum([0] + lengths)
        # Draw vertical lines and labels
        for i, b in enumerate(boundaries[:-1]):
            plt.axvline(x=b, color="white", linestyle="--", linewidth=1)

        # Add vowel labels
        midpoints = [
            (boundaries[i] + boundaries[i + 1]) // 2 for i in range(len(boundaries) - 1)
        ]
        for mid, label in zip(midpoints, labels):
            plt.text(mid, -5, label, ha="center", va="bottom", fontsize=8, rotation=45)

    # Optional: adjust ticks and layout
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()

    plt.show()


def load_encodings_from_path(folder, use_relu=False):

    encodings = []
    lengths = []
    labels = []

    current_time = 0

    for encoding_path in Path(folder).glob("*.pt"):
        encoding = load_encoding(encoding_path, use_relu=use_relu)
        T = encoding.shape[1]

        encodings.append(encoding)
        lengths.append(T)
        labels.append(encoding_path.stem)
        current_time += T

    encodings = np.concatenate(encodings, axis=1)
    return encodings, labels, lengths


def save_encodings_to_file(output_file, encodings, labels, lengths):
    with open(output_file, "wb") as f:
        pickle.dump(
            {
                "encodings": encodings,
                "labels": labels,
                "lengths": lengths,
            },
            f,
        )


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Plot Encoding")
    parser.add_argument(
        "--encoding_path",
        "-e",
        type=str,
        help="Path to the encoding file.",
        default="vowel_encodings",
    )
    parser.add_argument(
        "--relu",
        "-r",
        action="store_true",
        help="Apply ReLU activation to the encoding.",
        default=False,
    )
    args = parser.parse_args()

    encodings, labels, lengths = load_encodings_from_path(args.encoding_path, args.relu)

    relu_in_name = "_relu" if args.relu else ""
    save_encodings_to_file(
        Path(args.encoding_path) / f"vowel_encodings{relu_in_name}.pkl",
        encodings,
        labels,
        lengths,
    )
