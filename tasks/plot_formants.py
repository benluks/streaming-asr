from matplotlib import pyplot as plt
import pandas as pd
import torch
import torchaudio
import argparse


def plot_audio_and_formants(
    audio_file_path,
    formants_csv,
    time_interval=0.04,
    formants=[1, 2],
    plot_type="scatter",
):
    x, sr = torchaudio.load(audio_file_path)
    df = pd.read_csv(formants_csv, header=None)

    t_wave = torch.arange(x.shape[-1]) / sr

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 4]}
    )

    ax1.plot(t_wave, x[0].numpy(), label="Audio Signal")

    plot_function = getattr(ax2, plot_type)
    for formant in formants:
        plot_function(df[0], df[formant], label=f"F{formant}")

    ax2.set_ylabel("Formant Frequency (Hz)")
    ax2.set_title("Formants")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot audio waveform and formant tracks."
    )
    parser.add_argument("audio", help="Path or URL to audio file")
    parser.add_argument("formants_csv", help="Path to formant CSV file")
    parser.add_argument(
        "--formants",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Indices of formant columns to plot (e.g., 1 2 for F1 and F2)",
    )
    parser.add_argument(
        "--plot-type",
        choices=["plot", "scatter"],
        default="scatter",
        help="Type of plot to use for formants",
    )

    args = parser.parse_args()

    plot_audio_and_formants(
        audio_file_path=args.audio,
        formants_csv=args.formants_csv,
        time_interval=args.interval,
        formants=args.formants,
        plot_type=args.plot_type,
    )
