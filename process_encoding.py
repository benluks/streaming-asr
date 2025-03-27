import torch
from torch.nn.functional import relu
import matplotlib.pyplot as plt

def plot_encoding(encoding_path, use_relu=False):
    """
    Plot the encoding of a batch of audio chunks.
    Args:
        encoding_path (str): Path to the encoding file.
    """
    
    encoding = torch.load(f"{encoding_path}").squeeze(0)  # [16, 512]
    if use_relu:
        encoding = relu(encoding)
    encoding_np = encoding.T.numpy()

    plt.figure(figsize=(10, 6))
    plt.imshow(encoding_np, aspect='auto', cmap='magma', interpolation='nearest')
    plt.colorbar(label='Activation')
    plt.xlabel('Time step')
    plt.ylabel('Feature dimension')
    plt.title('Conformer Encoder Output')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Plot Encoding")
    parser.add_argument(
        "--encoding_path",
        "-e",
        type=str,
        help="Path to the encoding file.",
        default="vowel_encodings/i.pt",
    )
    parser.add_argument(
        "--relu",
        "-r",
        action="store_true",
        help="Apply ReLU activation to the encoding.",
        default=False,
    )
    args = parser.parse_args()
    plot_encoding(args.encoding_path, args.relu)