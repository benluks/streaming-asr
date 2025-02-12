import torch
import multiprocessing as mp
from typing import Generator


def stream_audio_file(
    file_path: str, chunk_size: int
) -> Generator[torch.Tensor, None, None]:
    """
    Streams an audio file in fixed-size chunks.

    Args:
        file_path (str): Path to the audio file.
        chunk_size (int): Number of frames per chunk.

    Yields:
        torch.Tensor: Audio chunk tensor.
    """
    from torchaudio.io import StreamReader

    streamer = StreamReader(file_path)
    yield from streamer.stream()


def create_device_stream(
    q: mp.Queue, format: str, src: str, segment_length: int, sample_rate: int
) -> None:
    """
    Captures audio from a device and pushes chunks into the queue.

    Args:
        q (mp.Queue): Queue for storing audio chunks.
        format (str): Audio format.
        src (str): Device source.
        segment_length (int): Number of frames per chunk.
    """
    from torchaudio.io import StreamReader

    print("Building stream reader...")
    try:
        streamer = StreamReader(src, format=format)
        streamer.add_basic_audio_stream(
            frames_per_chunk=segment_length, sample_rate=sample_rate, num_channels=1
        )

        for (chunk,) in streamer.stream(timeout=-1):
            try:
                q.put(chunk)
            except StopIteration:
                print("Streaming interrupted.")
                break
            except Exception as e:
                print(f"Error in streaming: {e}")
                break
    except KeyboardInterrupt:
        print("Streaming interrupted.")
    finally:
        print("Sending termination signal...")
        q.put(None)
