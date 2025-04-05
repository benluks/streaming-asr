import torch
from asr.inference import create_inference_process
from asr.streaming import create_device_stream
from utils import resolve_src

DEVICE = "avfoundation"
SRC=":4"
SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3"

CHUNK_FRAMES = 639
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 8
CHUNK_LEFT_CONTEXT = 2


def main(src, format, task="asr", vowel=None):
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
    output_q = manager.Queue()

    capture_process = ctx.Process(
        target=create_device_stream,
        args=(q, format, src, chunk_size_frames, MODEL_SAMPLE_RATE),
    )
    capture_process.start()

    inference_process = ctx.Process(target=create_inference_process, args=(q, task, output_q))
    inference_process.start()

    capture_process.join()
    inference_process.join()

    if task == "encode":
        encoding = torch.tensor([])
        while not output_q.empty():
            encoding = torch.cat((encoding, output_q.get()), dim=1)
        
        output_path = f"vowel_encodings/{vowel}.pt"
        print(f"Saving encoding to {output_path}")
        torch.save(encoding, output_path)
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time ASR from Microphone")
    parser.add_argument(
        "--src",
        "-s",
        type=str,
        help="Input source. Can be a file, URL, or device index (:[INT]).",
        default=":3",
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        help="Task to perform: 'asr' for transcription, 'encode' for feature extraction.",
        default="asr",
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

    src = SRC
    format = None

    from utils.vowel_files import vowel_files

    for vowel, vowel_path in vowel_files:    
        print(f"Encoding vowel: {vowel}")
        main(vowel_path, format, task="encode", vowel=vowel)
