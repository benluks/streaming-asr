import logging

logging.getLogger("speechbrain").setLevel(logging.WARNING)

import torch
from asr import transcribe, load_asr_model, transcribe_chunk
from streaming import create_device_stream
from utils import resolve_src
from decoder import create_decoding_process

DEVICE = "avfoundation"
SRC = ":3"
SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3"

CHUNK_FRAMES = 639
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 8
CHUNK_LEFT_CONTEXT = 2

chunk_len = CHUNK_SIZE * CHUNK_FRAMES


def create_inference_process(q, decoder_queue, chunk_size=CHUNK_SIZE):
    """
    Processes audio chunks from the queue and runs ASR or encoding.

    Args:
        q (mp.Queue): Queue containing audio chunks.
        mode (str): Either "asr" for transcription or "encode" for encoding.
    """
    asr_model, context = load_asr_model(CHUNK_SIZE, CHUNK_LEFT_CONTEXT)
    chunk_start = 0
    chunk_end = chunk_size

    print("Start speaking...")

    while True:
        chunk = q.get()
        if chunk is None:  # Exit condition
            break

        with torch.no_grad():
            chunk = chunk.squeeze(-1).unsqueeze(0).float()
            # factory_words = transcribe_chunk(asr_model, context, chunk)
            logits, words = transcribe(asr_model, context, chunk)

        decoder_queue.put(
            {"logits": logits, "words": words, "frames": (chunk_start, chunk_end)}
        )

        chunk_start = chunk_end
        chunk_end += chunk_size

        print(words, end="", flush=True)


def main(src, format):
    """
    Main function to initialize streaming and ASR processes.

    Args:
        mode (str): "asr" for full transcription or "encode" for feature extraction.
    """
    chunk_size_frames = CHUNK_FRAMES * CHUNK_SIZE

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    q = manager.Queue()
    decoding_queue = manager.Queue()

    capture_process = ctx.Process(
        target=create_device_stream,
        args=(q, format, src, chunk_size_frames, MODEL_SAMPLE_RATE),
    )
    capture_process.start()

    inference_process = ctx.Process(
        target=create_inference_process, args=(q, decoding_queue)
    )
    inference_process.start()

    decoding_process = ctx.Process(
        target=create_decoding_process, args=(decoding_queue, CHUNK_SIZE)
    )
    decoding_process.start()

    capture_process.join()
    inference_process.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ASR from Microphone")
    parser.add_argument(
        "--src",
        "-s",
        type=str,
        help="Input source. Can be a file, URL, or device index (:[INT]).",
        default=SAMPLE_FILE,
    )

    args = parser.parse_args()
    if args.src:
        src_type = resolve_src(args.src)
        if not src_type:
            raise ValueError("Invalid source type. ")
        else:
            src = args.src
            src = args.src
            format = DEVICE if src_type == "device" else None

    # src = SRC
    # format = DEVICE

    main(src, format)
