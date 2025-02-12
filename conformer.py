import torch
import torchaudio
import itertools
from speechbrain.inference.ASR import StreamingASR, ASRStreamingContext
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

import matplotlib.pyplot as plt


from vowel_files import vowel_files

# Initialize SpeechBrain's Streaming ASR Model

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

CHUNK_FRAMES = 639
MODEL_SAMPLE_RATE = 16000
CHUNK_SIZE = 8
CHUNK_LEFT_CONTEXT = 2


# -------------------------------
# 2. STREAMING & ENCODING FUNCTIONS
# -------------------------------


def stream_audio_file(file_path: str, chunk_size: int):
    """Streams an audio file in fixed-size chunks."""
    streamer = torchaudio.io.StreamReader(file_path)
    return asr_model._get_audio_stream(streamer, chunk_size)


def get_encoding(asr_model, context, chunk, chunk_len=None):
    if chunk_len is None:
        chunk_len = torch.ones((chunk.size(0),))

    chunk = chunk.float()
    chunk, chunk_len = chunk.to(asr_model.device), chunk_len.to(asr_model.device)

    encoding = asr_model.encode_chunk(context, chunk, chunk_len)
    return encoding


def encode_audio_stream(
    asr_model, file_path: str, context: ASRStreamingContext, decode=False
):
    """Streams and encodes an audio file, returning chunked encodings."""
    chunk_size_frames = asr_model.get_chunk_size_frames(context.config)

    rel_length = torch.tensor([1.0])
    final_chunks = [
        torch.zeros((1, chunk_size_frames), device=asr_model.device)
    ] * asr_model.hparams.fea_streaming_extractor.get_recommended_final_chunk_count(
        chunk_size_frames
    )

    chunks = stream_audio_file(file_path, chunk_size_frames)

    for chunk in itertools.chain(chunks, final_chunks):
        encoding = get_encoding(asr_model, context, chunk, rel_length)
        if decode:
            words, _ = asr_model.decode_chunk(context, encoding)
            yield words, encoding
        else:
            yield encoding


# -------------------------------
# 3. ENABLE DEVICE STREAMING
# -------------------------------


def create_device_stream(q, format, src, segment_length, sample_rate):
    from torchaudio.io import StreamReader

    print("Building StreamReader...")
    try:
        streamer = StreamReader(src, format=format)
        streamer.add_basic_audio_stream(
            frames_per_chunk=segment_length, sample_rate=sample_rate, num_channels=1
        )

        for (chunk,) in streamer.stream(timeout=-1):
            try:
                # print(f"Received chunk: {chunk.shape}")
                q.put(chunk)
            except StopIteration:
                print("Stream ended.")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
    except KeyboardInterrupt:
        print("Streaming interrupted.")
    finally:
        print("Sending termination signal...")
        q.put(None)  # Send shutdown signal


def create_inference_process(q):
    asr_model = StreamingASR.from_hparams(
        "speechbrain/asr-streaming-conformer-librispeech"
    )

    context = asr_model.make_streaming_context(
        DynChunkTrainConfig(CHUNK_SIZE, CHUNK_LEFT_CONTEXT)
    )

    print("Starting speaking...")

    while True:
        chunk = q.get()
        if chunk is None:  # Sentinel value to stop the process
            break
        
        output = ""

        chunk = chunk.squeeze(-1).unsqueeze(0)
        words = asr_model.transcribe_chunk(context, chunk)
        output += words[0]
        
        print(output, end="", flush=True)


def main(device="avfoundation", src=":3"):
    
    chunk_size_frames = CHUNK_FRAMES * CHUNK_SIZE
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    q = manager.Queue()

    capture_process = ctx.Process(
        target=create_device_stream, args=(q, device, src, chunk_size_frames, MODEL_SAMPLE_RATE)
    )
    capture_process.start()

    inference_process = ctx.Process(target=create_inference_process, args=(q,))
    inference_process.start()

    capture_process.join()
    inference_process.join()


# asr_model = StreamingASR.from_hparams("speechbrain/asr-streaming-conformer-librispeech")

# model_sr = asr_model.hparams.sample_rate
# context = asr_model.make_streaming_context(
#     DynChunkTrainConfig(CHUNK_SIZE, CHUNK_LEFT_CONTEXT)
# )


if __name__ == "__main__":
    main()

    # vowel, vowel_path = random.choice(vowel_files)
    # print(vowel)
    # for encoding in encode_audio_stream(asr_model, vowel_path, context):
    #     words, _ = asr_model.decode_chunk(context, encoding)
    #     if words[0]:
    #         print(words[0])
