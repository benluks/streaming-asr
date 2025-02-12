import torch
from speechbrain.inference.ASR import StreamingASR, ASRStreamingContext
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

CHUNK_SIZE = 8  # Adjust for different chunk durations
CHUNK_LEFT_CONTEXT = 2  # Number of previous chunks used as context


def load_asr_model() -> StreamingASR:
    """
    Loads the SpeechBrain ASR streaming model.

    Returns:
        StreamingASR: Initialized ASR model.
    """
    return StreamingASR.from_hparams("speechbrain/asr-streaming-conformer-librispeech")


def get_encoding(
    asr_model: StreamingASR, context: ASRStreamingContext, chunk: torch.Tensor
) -> torch.Tensor:
    """
    Encodes a chunk of audio.

    Args:
        asr_model (StreamingASR): The ASR model instance.
        context (ASRStreamingContext): The streaming context.
        chunk (torch.Tensor): The audio chunk to encode.

    Returns:
        torch.Tensor: Encoded representation of the audio chunk.
    """
    if chunk_len is None:
        chunk_len = torch.ones((chunk.size(0),))

    chunk = chunk.float()

    return asr_model.encode_chunk(context, chunk, chunk_len)


def transcribe_chunk(
    asr_model: StreamingASR, context: ASRStreamingContext, chunk: torch.Tensor
) -> str:
    """
    Transcribes an audio chunk using ASR.

    Args:
        asr_model (StreamingASR): The ASR model instance.
        context (ASRStreamingContext): The streaming context.
        chunk (torch.Tensor): The audio chunk.

    Returns:
        str: Transcribed text.
    """
    words = asr_model.transcribe_chunk(context, chunk)
    return words[0]
