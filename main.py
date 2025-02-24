import torch
from asr import load_asr_model, run_dummy_inference
from streaming import create_device_stream
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from utils import resolve_src

DEVICE = "avfoundation"
SRC = ":4"
SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3"

CHUNK_FRAMES = 639
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 8
CHUNK_LEFT_CONTEXT = 2

RAVE_MODEL_PATH = "models/musicnet.ts"
ASR_MODEL_PATH = "speechbrain/asr-streaming-conformer-librispeech"


def create_playback_process(q, model_sample_rate, chunk_size, model_channels):
    """
    Processes audio chunks from the queue and plays them back.

    Args:
        q (mp.Queue): Queue containing audio chunks.
    """
    import sounddevice as sd

    # from playback import audio_callback
    import queue

    audio_queue = queue.Queue(maxsize=10)

    def audio_callback(outdata, frames, time, status):
        if status:
            print(status)
        try:
            chunk = audio_queue.get_nowait()  # Get latest chunk
        except queue.Empty:
            chunk = torch.zeros((frames, model_channels), dtype=torch.float32)
        outdata[:] = chunk.reshape(-1, model_channels)

    with sd.OutputStream(
        samplerate=model_sample_rate,
        blocksize=chunk_size,
        channels=model_channels,
        callback=audio_callback,
        dtype="float32",
    ) as out_stream:
        print("Playing audio...")
        while True:
            try:
                chunk = q.get()
                audio_queue.put(chunk)
            except KeyboardInterrupt:
                break


def create_inference_process(inference_queue, mode, model_path, playback_queue):
    """
    Processes audio chunks from the queue and runs ASR or encoding.

    Args:
        q (mp.Queue): Queue containing audio chunks.
        mode (str): Either "asr" for transcription or "encode" for encoding.
    """

    if mode == "asr":
        model = load_asr_model(model_path)
        context = model.make_streaming_context(
            DynChunkTrainConfig(CHUNK_SIZE, CHUNK_LEFT_CONTEXT)
        )
        print("Start speaking...")

    elif mode == "rave":
        model = load_asr_model(model_path)

    elif mode == "dummy":
        model = None

    while True:
        chunk = inference_queue.get()
        if chunk is None:  # Exit condition
            break

        chunk = chunk.squeeze(-1).unsqueeze(0)

        # run appropriate inference
        if mode == "asr":
            words = model.transcribe_chunk(context, chunk)
            print(words[0], end="", flush=True)
        else:
            output = run_dummy_inference(chunk)
            playback_queue.put(output)


def main(src, format, output, task, model_path, model_sr, model_channels):
    """
    Main function to initialize streaming and ASR processes.

    Args:
        mode (str): "asr" for full transcription or "encode" for feature extraction.
    """

    if task == "asr":
        chunk_size_frames = CHUNK_FRAMES * CHUNK_SIZE
    else:
        chunk_size_frames = 256

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    inference_queue = manager.Queue()

    model_channels = 1 if task == "asr" else 2

    capture_process = ctx.Process(
        target=create_device_stream,
        args=(
            inference_queue,
            format,
            src,
            chunk_size_frames,
            model_sr,
            model_channels,
        ),
    )
    capture_process.start()

    if output:
        playback_queue = manager.Queue()
        playback_process = ctx.Process(
            target=create_playback_process,
            args=(playback_queue, model_sr, chunk_size_frames, model_channels),
        )
        playback_process.start()
    else:
        playback_queue = None

    inference_process = ctx.Process(
        target=create_inference_process,
        args=(inference_queue, task, model_path, playback_queue),
    )
    inference_process.start()

    capture_process.join()
    inference_process.join()
    if output:
        playback_process.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ASR from Microphone")

    # parser.add_argument(
    #     "--test",
    #     "-t",
    #     action="store_true",
    #     help="Test audio playback.",
    # )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="Task to run. Can be 'asr', 'rave', or 'dummy'.",
        default="dummy",
    )

    parser.add_argument(
        "--src",
        "-s",
        type=str,
        help="Input source. Can be a file, URL, or device index (:[INT]).",
        default=":3",
    )

    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="""Path to the model file
        (ASR: speechbrain/asr-streaming-conformer-librispeech, RAVE: models/musicnet.ts).""",
        default=None,
    )

    parser.add_argument(
        "--output",
        "-o",
        action="store_true",
        help="Output audio to speakers.",
    )

    args = parser.parse_args()

    if args.task == "dummy":
        args.src = ":3"
        args.output = True
        model_sr = 44100
        model_channels = 2

    if args.src:
        src_type = resolve_src(args.src)
        if not src_type:
            raise ValueError("Invalid source type. ")
        else:
            src = args.src
            format = DEVICE if src_type == "device" else None

    # src = SRC
    # format = DEVICE

    if not args.model_path:
        args.model_path = ASR_MODEL_PATH if args.task == "asr" else RAVE_MODEL_PATH

    main(src, format, args.output, args.task, args.model_path, model_sr, model_channels)
