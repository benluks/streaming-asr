from asr import load_asr_model
from streaming import create_device_stream
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

DEVICE = "avfoundation"
SRC=":3"
SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3"

CHUNK_FRAMES = 639
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 8
CHUNK_LEFT_CONTEXT = 2


def create_inference_process(q, mode="asr"):
    """
    Processes audio chunks from the queue and runs ASR or encoding.

    Args:
        q (mp.Queue): Queue containing audio chunks.
        mode (str): Either "asr" for transcription or "encode" for encoding.
    """
    asr_model = load_asr_model()
    context = asr_model.make_streaming_context(
        DynChunkTrainConfig(CHUNK_SIZE, CHUNK_LEFT_CONTEXT)
    )

    print("Start speaking...")

    while True:
        chunk = q.get()
        if chunk is None:  # Exit condition
            break

        chunk = chunk.squeeze(-1).unsqueeze(0)
        words = asr_model.transcribe_chunk(context, chunk)

        print(words[0], end="", flush=True)


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

    capture_process = ctx.Process(
        target=create_device_stream,
        args=(q, format, src, chunk_size_frames, MODEL_SAMPLE_RATE),
    )
    capture_process.start()

    inference_process = ctx.Process(target=create_inference_process, args=(q,))
    inference_process.start()

    capture_process.join()
    inference_process.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ASR from Microphone")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="Path to an audio file for transcription. If omitted, defaults to live microphone input.",
    )

    args = parser.parse_args()
    if args.file:
        src = args.file
        format = None
    else:
        src = SRC
        format = DEVICE

    main(src, format)
