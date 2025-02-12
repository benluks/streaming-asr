from asr import load_asr_model
from streaming import create_device_stream
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

DEVICE = "avfoundation"
SRC = ":3"

CHUNK_FRAMES = 639
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 8
CHUNK_LEFT_CONTEXT = 2


def create_inference_process(q):
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

    print("Starting speaking...")

    while True:
        chunk = q.get()
        if chunk is None:  # Exit condition
            break

        output = ""

        chunk = chunk.squeeze(-1).unsqueeze(0)
        words = asr_model.transcribe_chunk(context, chunk)
        output += words[0]

        print(output, end="", flush=True)


def main():
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
        args=(q, DEVICE, SRC, chunk_size_frames, MODEL_SAMPLE_RATE),
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
        "--mode",
        choices=["asr", "encode"],
        default="asr",
        help="Choose ASR (transcription) or encoding mode",
    )

    args = parser.parse_args()
    main()
