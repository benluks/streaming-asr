import torch
from asr import load_asr_model, get_encoding
from streaming import create_device_stream
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from utils import resolve_src

DEVICE = "avfoundation"
SRC=":4"
SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3"

CHUNK_FRAMES = 639
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 8
CHUNK_LEFT_CONTEXT = 2


def create_inference_process(q, task, output_q=None):
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

    encoding = torch.tensor([])


    while True:
        chunk = q.get()
        if chunk is None:  # Exit condition
            break

        chunk = chunk.squeeze(-1).unsqueeze(0)
        if task == "asr":
            words = asr_model.transcribe_chunk(context, chunk)
            print(words[0], end="", flush=True)
        elif task == "encode":
            output = get_encoding(asr_model, context, chunk)
            if output_q:
                output_q.put(output)



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

    from vowel_files import vowel_files

    for vowel, vowel_path in vowel_files:    
        print(f"Encoding vowel: {vowel}")
        main(vowel_path, format, task="encode", vowel=vowel)
