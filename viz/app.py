# viz/app.py

import pandas as pd
import streamlit as st
import torch
import numpy as np
import plotly.express as px
from pathlib import Path
import sys

import torchaudio
from utils.vowel_files import vowel_files_dict

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
def run_pca_on_all_layers(*args):
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


@st.cache_data
def load_vowel_audio(vowel, t_pca):
    waveform, sr = torchaudio.load(vowel_files_dict[vowel])
    waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze().numpy()

    resampled = np.interp(
        np.linspace(0, len(waveform), t_pca), np.arange(len(waveform)), waveform
    )
    return resampled


# === UI ===
st.title("Transformer Layer Explorer")

encoding_path = st.text_input(
    "Path to encoding dict:", "vowel_encodings/encodings_hidden_states.pkl"
)


if Path(encoding_path).exists():

    encodings, labels, lengths, boundaries = load_encodings(encoding_path)
    pcas, transformed, cumulatives = run_pca_on_all_layers(encodings)

    vowels = st.multiselect("Select vowel(s)", list(labels), default=["a", "i", "u"])
    selected_vowel_ids = [labels.index(vowel) for vowel in vowels]

    layer = st.slider("Layer", 0, len(transformed) - 1, 0)

    explained = cumulatives[layer]
    explained_90 = explained[explained < 0.9]

    # pc = st.slider("Component", 0, explained_90.size - 1, 0)
    pcs = st.multiselect("PCs", list(range(explained_90.size)), default=[0, 1, 2])

    with st.expander(f"Layer {layer}, PCs Plot", expanded=True):

        # Concatenate selected vowel segments across time for all PCs
        segments = [
            transformed[layer][boundaries[i] : boundaries[i + 1], pcs]
            for i in selected_vowel_ids
        ]
        X = np.vstack(segments)  # shape [T_total, num_pcs]

        # Melt to long format for Plotly
        df = pd.DataFrame(X, columns=[f"PC{i}" for i in pcs])
        df["Time"] = df.index
        df_melted = df.melt(id_vars="Time", var_name="Component", value_name="Value")

        # Plot multiple PC tracks
        fig = px.line(
            df_melted,
            x="Time",
            y="Value",
            color="Component",
            title=f"Layer {layer}: PCs {','.join(map(str, pcs))}",
        )

        # fig = px.line(y=selected_segments)
        cumulative_end_ids = np.cumsum([0] + [lengths[i] for i in selected_vowel_ids])
        midpoints = cumulative_end_ids[:-1] + (
            (cumulative_end_ids[1:] - cumulative_end_ids[:-1]) // 2
        )

        for vowel_label, end_id, midpoint in zip(
            vowels, cumulative_end_ids[1:], midpoints
        ):
            fig.add_vline(x=end_id, line_dash="dash", line_color="green")
            fig.add_annotation(
                x=midpoint,
                y=X[:, range(len(pcs))].max() + 5,
                text=vowel_label,
                showarrow=False,
                # yshift=-100,  # Move label below axis (adjust as needed)
                font=dict(size=14),
            )

        st.plotly_chart(fig, use_container_width=True)

        cols = st.columns(len(vowels))
        for col, vowel in zip(cols, vowels):
            wav, sr = torchaudio.load(vowel_files_dict[vowel])
            with col:
                st.markdown(
                    f"<div style='text-align: center; font-weight: bold'>{vowel}</div>",
                    unsafe_allow_html=True,
                )
                st.audio(wav[0].numpy(), sample_rate=sr)

    with st.expander(f"Layer {layer}, PCs Plot", expanded=True):
        fig2 = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_90))],
            y=explained_90,
            labels={"x": "PC", "y": "Explained Variance"},
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.warning("Encoding file not found.")
