from pathlib import Path
import pickle
import time
import numpy as np
import torch
from torch.nn.functional import relu
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from asr.asr import load_asr_model, batch_encode
from vowel_files import vowel_files

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
    parser.add_argument(
        "--only_keep_vowels",
        "-k",
        action="store_true",
        help="Only include common pulmonic vowels (listed in `KEEP_VOWELS`).",
        default=False,
    )
    args = parser.parse_args()

    asr_model, context = load_asr_model()
    encodings = {}

    for vowel_label, vowel_path in vowel_files:
        print(f"Encoding {vowel_label}")

        processed = False
        while not processed:
            try:
                output, hidden_states = batch_encode(
                    asr_model, context, vowel_path, output_hidden_states=True
                )
                processed = True
            except RuntimeError:
                print(
                    f"Internal server error while fetching {vowel_label}.\nTrying again in 10 seconds..."
                )
                time.sleep(10)

        encodings[vowel_label] = {
            "output": output,
            "hidden_states": hidden_states,
        }

    torch.save(encodings, "encodings_hidden_states.pt")
