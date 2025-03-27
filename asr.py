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
    asr_model: StreamingASR,
    context: ASRStreamingContext,
    chunk: torch.Tensor,
    chunk_len=None,
):
    """Encoding of a batch of audio chunks into a batch of encoded
    sequences.
    For full speech-to-text offline transcription, use `transcribe_batch` or
    `transcribe_file`.
    Must be called over a given context in the correct order of chunks over
    time.

    Arguments
    ---------
    context : ASRStreamingContext
        Mutable streaming context object, which must be specified and reused
        across calls when streaming.
        You can obtain an initial context by calling
        `asr.make_streaming_context(config)`.

    chunk : torch.Tensor
        The tensor for an audio chunk of shape `[batch size, time]`.
        The time dimension must strictly match
        `asr.get_chunk_size_frames(config)`.
        The waveform is expected to be in the model's expected format (i.e.
        the sampling rate must be correct).

    chunk_len : torch.Tensor, optional
        The relative chunk length tensor of shape `[batch size]`. This is to
        be used when the audio in one of the chunks of the batch is ending
        within this chunk.
        If unspecified, equivalent to `torch.ones((batch_size,))`.

    Returns
    -------
    torch.Tensor
        Encoded output, of a model-dependent shape."""

    if chunk_len is None:
        chunk_len = torch.ones((chunk.size(0),))

    chunk = chunk.float()
    # chunk, chunk_len = chunk.to(self.device), chunk_len.to(self.device)

    assert chunk.shape[-1] <= asr_model.get_chunk_size_frames(context.config)

    x = asr_model.hparams.fea_streaming_extractor(
        chunk, context=context.fea_extractor_context, lengths=chunk_len
    )
    x = asr_model.mods.enc.forward_streaming(x, context.encoder_context)

    return x


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
