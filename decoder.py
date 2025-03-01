from pyctcdecode import build_ctcdecoder
import torch

asr_model = None


def get_vocabulary(tokenizer):
    tokenizer = asr_model.hparams.tokenizer
    return tokenizer.decode([[i] for i in range(tokenizer.vocab_size())])


def build_decoder(
    vocabulary,
    kenlm_model_path="kenlm/3-gram.pruned.1e-7.arpa",
    alpha=0.5,
    beta=1.0,
):
    vocabulary = get_vocabulary(asr_model)
    decoder = build_ctcdecoder(vocabulary, kenlm_model_path, alpha, beta)
    return decoder


def create_decoding_process(decoding_queue, vocabulary, buffer_len=128):
    """
    Processes encodings from the queue and runs decoding.

    Args:
        decoding_queue (mp.Queue): Queue containing encodings.
    """
    decoder = build_decoder(vocabulary)

    print("Start decoding...")

    buffer = torch.tensor([])

    while True:
        logits = decoding_queue.get()
        # [1, t, v]
        torch.cat((buffer, logits))[:, -(min(buffer_len, buffer.size(1))) :, :]
        if logits is None:  # Exit condition
            break

        words = decoder.decode(buffer)
        # print(words, end="", flush=True)
