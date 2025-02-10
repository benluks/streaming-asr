import torch
import torchaudio
import itertools
from speechbrain.inference.ASR import StreamingASR, ASRStreamingContext
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig
from IPython.display import Audio
import random
import matplotlib.pyplot as plt


from vowel_files import vowel_files

# Initialize SpeechBrain's Streaming ASR Model

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

CHUNK_SIZE = 2
CHUNK_LEFT_CONTEXT = 4

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


asr_model = StreamingASR.from_hparams("speechbrain/asr-streaming-conformer-librispeech")
context = asr_model.make_streaming_context(
    DynChunkTrainConfig(CHUNK_SIZE, CHUNK_LEFT_CONTEXT)
)


if __name__ == "__main__":
    vowel, vowel_path = random.choice(vowel_files)
    print(vowel)
    for encoding in encode_audio_stream(asr_model, vowel_path, context):
        words, _ = asr_model.decode_chunk(context, encoding)
        if words[0]:
            print(words[0])
