from pathlib import Path
from pyctcdecode import build_ctcdecoder
import torch

from utils import read_vocab_file

asr_model = None


def get_vocabulary(tokenizer):
    tokenizer = asr_model.hparams.tokenizer
    return tokenizer.decode([[i] for i in range(tokenizer.vocab_size())])


def build_decoder(
    vocabulary,
    kenlm_model_path="lm/3-gram.pruned.1e-7.binary",
    unigrams_path="lm/unigrams.txt",
    alpha=0.5,
    beta=1.0,
):

    if Path(vocabulary).is_file():
        labels = read_vocab_file(vocabulary)
    else:
        raise FileNotFoundError(f"Vocabulary file not found: {vocabulary}")

    if Path(unigrams_path).is_file():
        unigrams = read_vocab_file(unigrams_path)

    decoder = build_ctcdecoder(labels, kenlm_model_path, unigrams=unigrams)
    return decoder


def create_decoding_process(
    decoding_queue, chunk_frames, vocabulary="lm/vocab.txt", buffer_len=128
):
    """
    Processes encodings from the queue and runs decoding.

    Args:
        decoding_queue (mp.Queue): Queue containing encodings.
    """

    assert (
        buffer_len % chunk_frames == 0
    ), "`buffer_len` must be a multiple of `chunk_frames`"
    
    decoder = build_decoder(vocabulary)

    print("Start decoding...")

    buffer = torch.tensor([])
    second_pass_start = 0
    second_pass_end = None

    while True:
        first_pass_output = decoding_queue.get()
        logits = first_pass_output.get("logits")
        if logits is None:  # Exit condition
            break

        buffer = torch.cat((buffer, logits[0]))
        buffer = buffer[-(min(buffer_len, buffer.shape[0])) :]

        words = decoder.decode(buffer.numpy())
        print(words, end="", flush=True)
