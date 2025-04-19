import os
from pathlib import Path
import pickle

import numpy as np
import torch
import urllib


def load_pickle(file_path):

    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_encodings_from_pkl(encodings_path):

    data = load_pickle(encodings_path)

    X = data["encodings"]

    if X.ndim == 2:
        X = X.T

    labels = data["labels"]
    lengths = data["lengths"]

    boundaries = np.cumsum([0] + lengths)

    return X, labels, lengths, boundaries


def create_encoding_dict(encoding_path):

    X, labels, lengths = load_encodings_from_pkl(encoding_path)

    # Create a dictionary to store the encodings
    encoding_dict = {}
    start_idx = 0
    for label, length in zip(labels, lengths):
        encoding_dict[label] = X[start_idx : start_idx + length]
        start_idx += length
    return encoding_dict


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


# INDIVIDUAL VOWEL ENCODINGS (.pt FORMAT)


def load_individual_vowel_encoding(encoding_path, use_relu=False):
    """
    Encodings for individual vowels are stored at .pt files, whereas the full encodings are stored as `.pickle`d numpy arrays.
    """
    encoding = torch.load(f"{encoding_path}").squeeze(0)
    if use_relu:
        encoding = torch.nn.functional.relu(encoding)
    return encoding.T.numpy()


def load_and_concatenate_encodings(
    encodings_dir, use_relu=False, only_keep_vowels=False
):

    encodings = []
    lengths = []
    labels = []

    for encoding_path in Path(encodings_dir).glob("*.pt"):
        if only_keep_vowels and (encoding_path.stem not in KEEP_VOWELS):
            continue
        encoding = load_individual_vowel_encoding(encoding_path, use_relu=use_relu)
        T = encoding.shape[1]

        encodings.append(encoding)
        lengths.append(T)
        labels.append(encoding_path.stem)

    encodings = np.concatenate(encodings, axis=1)
    return encodings, labels, lengths


def save_pickle(output_file, object: dict):
    """
    Save the encodings to a file.
    Args:
        output_file (str): Path to the output file.
        object (dict): Dictionary containing encodings indexed by vowel labels.
    """
    with open(output_file, "wb") as f:
        pickle.dump(object, f)


def save_encodings_to_file(output_file, encodings, labels, lengths):
    save_pickle(
        output_file,
        {
            "encodings": encodings,
            "labels": labels,
            "lengths": lengths,
        },
    )


is_device = lambda src: src.startswith(":") and src[1:].isdigit()


def resolve_src(src):

    if is_device(src):
        return "device"

    src = src.strip()
    # Check if it's a local file
    if os.path.exists(src) or Path(src).exists():
        return "file"

    parsed = urllib.parse.urlparse(src)
    if parsed.scheme in ("http", "https", "ftp"):
        return "url"

    return None
