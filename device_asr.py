"""
Device ASR with Emformer RNN-T
==============================

**Author**: `Moto Hira <moto@meta.com>`__, `Jeff Hwang <jeffhwang@meta.com>`__.

This tutorial shows how to use Emformer RNN-T and streaming API
to perform speech recognition on a streaming device input, i.e. microphone
on laptop.
"""

import torch
import torchaudio
import torchaudio.functional as F


def save_audio(chunks, filename, sample_rate):
    """Save the collected audio chunks into a WAV file."""
    audio = torch.cat(chunks, dim=0)  # Concatenate all recorded chunks
    audio = F.resample(audio, orig_freq=sample_rate, new_freq=16000)  # Downsample if needed
    torchaudio.save(filename, audio.T, 16000, format="wav")  # Save as 16kHz mono

# The data acquisition process will stop after this number of steps.
# This eliminates the need of process synchronization and makes this
# tutorial simple.
NUM_ITER = 100


def stream(q, format, src, segment_length, sample_rate, save_path="streamed_audio.wav"):
    from torchaudio.io import StreamReader

    print("Building StreamReader...")
    streamer = StreamReader(src, format=format)
    streamer.add_basic_audio_stream(frames_per_chunk=segment_length, sample_rate=sample_rate)

    print(streamer.get_src_stream_info(0))
    print(streamer.get_out_stream_info(0))

    print("Streaming...")
    print()
    
    stream_iterator = streamer.stream()

    recorded_chunks = []

    for _ in range(NUM_ITER):
        (chunk,) = next(stream_iterator)
        q.put(chunk)
        recorded_chunks.append(chunk)

    print(f"Saving recorded chunks to {save_path}...")
    save_audio(recorded_chunks, save_path, sample_rate)

######################################################################

class Pipeline:
    """Build inference pipeline from RNNTBundle.

    Args:
        bundle (torchaudio.pipelines.RNNTBundle): Bundle object
        beam_width (int): Beam size of beam search decoder.
    """

    def __init__(self, bundle: torchaudio.pipelines.RNNTBundle, beam_width: int = 10):
        self.bundle = bundle
        self.feature_extractor = bundle.get_streaming_feature_extractor()
        self.decoder = bundle.get_decoder()
        self.token_processor = bundle.get_token_processor()

        self.beam_width = beam_width

        self.state = None
        self.hypotheses = None

    def infer(self, segment: torch.Tensor) -> str:
        """Perform streaming inference"""
        features, length = self.feature_extractor(segment)
        self.hypotheses, self.state = self.decoder.infer(
            features, length, self.beam_width, state=self.state, hypothesis=self.hypotheses
        )
        transcript = self.token_processor(self.hypotheses[0][0], lstrip=False)
        return transcript


######################################################################
#


class ContextCacher:
    """Cache the end of input data and prepend the next input data with it.

    Args:
        segment_length (int): The size of main segment.
            If the incoming segment is shorter, then the segment is padded.
        context_length (int): The size of the context, cached and appended.
    """

    def __init__(self, segment_length: int, context_length: int):
        self.segment_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.segment_length:
            chunk = torch.nn.functional.pad(chunk, (0, self.segment_length - chunk.size(0)))
        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]
        return chunk_with_context


######################################################################
# The main process
# -------------------


def main(device, src, bundle):
    print(torch.__version__)
    print(torchaudio.__version__)

    print("Building pipeline...")
    pipeline = Pipeline(bundle)

    sample_rate = bundle.sample_rate
    segment_length = bundle.segment_length * bundle.hop_length
    context_length = bundle.right_context_length * bundle.hop_length

    print(f"Sample rate: {sample_rate}")
    print(f"Main segment: {segment_length} frames ({segment_length / sample_rate} seconds)")
    print(f"Right context: {context_length} frames ({context_length / sample_rate} seconds)")

    cacher = ContextCacher(segment_length, context_length)

    @torch.inference_mode()
    def infer():
        for _ in range(NUM_ITER):
            chunk = q.get()
            segment = cacher(chunk[:, 0])
            transcript = pipeline.infer(segment)
            print(transcript, end="\r", flush=True)
            

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=stream, args=(q, device, src, segment_length, sample_rate))
    p.start()
    infer()
    p.join()
    
        


if __name__ == "__main__":
    main(
        device="avfoundation",
        src=":3",
        bundle=torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH,
    )
