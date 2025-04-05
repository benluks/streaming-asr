# tasks/encode_batch.py

from pathlib import Path
import time
import argparse
import torch
from asr.asr import load_asr_model, batch_encode
from vowel_files import vowel_files
from utils.io import KEEP_VOWELS


def run_batch_encoding(output_path, relu=False, only_keep_vowels=False):
    asr_model, context = load_asr_model()
    encodings = {}

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for vowel_label, vowel_path in vowel_files:
        if only_keep_vowels and (vowel_label not in KEEP_VOWELS):
            continue

        print(f"Encoding {vowel_label}")
        processed = False
        while not processed:
            try:
                output, hidden_states = batch_encode(
                    asr_model, context, vowel_path, output_hidden_states=True
                )
                processed = True
            except RuntimeError:
                print(f"Error fetching {vowel_label}, retrying in 10s...")
                time.sleep(10)

        if relu:
            output = torch.relu(output)
            hidden_states = [torch.relu(h) for h in hidden_states]

        encodings[vowel_label] = {
            "output": output,
            "hidden_states": hidden_states,
        }

    print(f"Saving encodings to {output_path}")
    torch.save(encodings, output_path)
    return encodings


def encoding_dict_to_full_tensor(encoding_dict, only_keep_vowels=False, use_relu=False):
    """
    returns encodings as nummpy with shape [B, D, T]
    """
    labels = []
    lengths = []
    encodings = torch.tensor([])

    vowel_ref = KEEP_VOWELS if only_keep_vowels else encoding_dict

    for key in vowel_ref:

        labels.append(key)

        hidden_states = torch.cat(encoding_dict[key]["hidden_states"])
        output = encoding_dict[key]["output"]

        lengths.append(hidden_states.shape[1])
        vowel_encodings = torch.cat([hidden_states, output])
        encodings = torch.cat([encodings, vowel_encodings], dim=1)

    return encodings.transpose(1, 2).numpy(), labels, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Encode Vowel Files")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="vowel_encodings/encodings_hidden_states_dict.pt",
    )
    parser.add_argument(
        "--full_encoding",
        "-f",
        action="store_true",
        help="Save encodings as a full tensor of `encdoings`, `labels`, and `lengths`",
    )
    parser.add_argument(
        "--encoding_dict_path",
        "-d",
        type=str,
        default="vowel_encodings/encodings_hidden_states_dict.pt",
        help="Path to the encoding dictionary to convert to full tensor. Only used if --full_encoding is set.",
    )
    parser.add_argument(
        "--hidden_states",
        "-hs",
        action="store_true",
        help="Save hidden states in addition to final output",
    )
    parser.add_argument(
        "--relu", "-r", action="store_true", help="Apply ReLU to encodings"
    )
    parser.add_argument("--only_keep_vowels", "-k", action="store_true")

    args = parser.parse_args()

    if not args.full_encoding:
        run_batch_encoding(args.output, args.relu, args.only_keep_vowels)
    else:
        if not Path(args.encoding_dict_path).exists():
            print("doesn't exist")
            # just for now
            exit()
            encoding_dict = run_batch_encoding(
                args.output, args.relu, args.only_keep_vowels
            )
        else:
            encoding_dict = torch.load(args.encoding_dict_path)
            print(f"Loaded encodings from {args.encoding_dict_path}")

        encodings, labels, lengths = encoding_dict_to_full_tensor(
            encoding_dict,
            only_keep_vowels=args.only_keep_vowels,
            use_relu=args.relu,
        )
        from utils.io import save_pickle
        
        save_pickle(
            args.output,
            {
                "encodings": encodings,
                "labels": labels,
                "lengths": lengths,
            },
        )
        print(f"Saved encodings to {args.output}")
