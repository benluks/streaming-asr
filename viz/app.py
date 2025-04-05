# viz/app.py

import streamlit as st
import torch
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.io import load_encodings_from_pkl
from utils.stats import run_pca


@st.cache_data
def load_encodings(path):
    """
    returns: encodings, labels, lengths
    encodings: [N, D, T] -> [num layers=14, model_d=512, time_step=752]

    """
    return load_encodings_from_pkl(path)


@st.cache_data
def run_pca_per_layer(*args):
    """
    Run PCA on each layer of the encodings.
    Args:
        encodings (torch.Tensor): tensor of shape [num_hidden_layers, model_d, time_step].
        labels (list): List vowel encoded.
        lengths (list): Lengths of each encoded vowel.
    Returns:
        transformed (list): List of PCA-transformed encodings.
        pcs (list): List of PCA objects for each layer.
        cumulative_explained_variance (list): Cumulative explained variance for each layer.
    """
    encodings = args[0]

    pcas = []
    transformed = []
    cumulatives = []

    for layer in encodings:
        layer = layer.T  # [T, D]
        pca, cumulative = run_pca(layer)

        pcas.append(pca)
        transformed.append(pca.transform(layer))
        cumulatives.append(cumulative)

    return pcas, transformed, cumulatives


# === UI ===
st.title("Transformer Layer Explorer")

encoding_path = st.text_input(
    "Path to encoding dict:", "vowel_encodings/encodings_hidden_states.pkl"
)


if Path(encoding_path).exists():
    encodings, labels, lengths = load_encodings(encoding_path)

    vowel = st.selectbox("Choose vowel", list(labels))
    pcas, transformed, cumulatives = run_pca_per_layer(encodings, vowel)

    layer = st.slider("Layer", 0, len(transformed) - 1, 0)

    explained = cumulatives[layer]
    explained_90 = explained[explained < 0.9]

    pc = st.slider("Component", 0, explained_90.size - 1, 0)

    with st.expander(f"Layer {layer}, PC {pc} Plot", expanded=True):
        fig = px.line(y=transformed[layer][:, pc])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander(f"Layer {layer}, PC {pc} Plot", expanded=True):
        fig2 = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_90))],
            y=explained_90,
            labels={"x": "PC", "y": "Explained Variance"},
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("Encoding file not found.")
