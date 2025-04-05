# tasks/encode_batch.py

import time
import argparse
import torch
from asr.asr import load_asr_model, batch_encode
from vowel_files import vowel_files
from utils.encoding_utils import KEEP_VOWELS


def run_batch_encoding(output_path, relu=False, only_keep_vowels=False):
    asr_model, context = load_asr_model()
    encodings = {}

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Encode Vowel Files")
    parser.add_argument(
        "--output", "-o", type=str, default="encodings_hidden_states.pt"
    )
    parser.add_argument(
        "--relu", "-r", action="store_true", help="Apply ReLU to encodings"
    )
    parser.add_argument("--only_keep_vowels", "-k", action="store_true")

    args = parser.parse_args()
    run_batch_encoding(args.output, args.relu, args.only_keep_vowels)
