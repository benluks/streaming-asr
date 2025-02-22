import sounddevice as sd
import torch
from torchaudio.io import StreamReader
import queue
from argparse import ArgumentParser

SAMPLE_RATE = 44100
CHUNK_SIZE = 1024
CHANNELS = 1

SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3"
SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/e/e5/%22The_Storming_of_El_Caney%22_by_Russell_Alexander.wav"

audio_queue = queue.Queue(maxsize=10)


def audio_callback(outdata, frames, time, status):
    if status:
        print(status)
    try:
        chunk = audio_queue.get_nowait()  # Get latest chunk
    except queue.Empty:
        chunk = torch.zeros((frames, CHANNELS), dtype=torch.float32)
    outdata[:] = chunk.reshape(-1, CHANNELS)


def main(file_path):

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=CHANNELS,
        callback=audio_callback,
    ):
        print("Streaming audio...")
        streamer = StreamReader(file_path)
        streamer.add_basic_audio_stream(
            frames_per_chunk=CHUNK_SIZE, num_channels=CHANNELS, sample_rate=SAMPLE_RATE
        )
        for (chunk,) in streamer.stream(timeout=-1):
            audio_queue.put(chunk)


if __name__ == "__main__":

    # parser = ArgumentParser(description="Real-time output streaming from file")
    # parser.add_argument(
    #     "file_path",
    #     "-f",
    #     type=str,
    #     required=False,
    #     help="Path to the audio file",
    #     default=SAMPLE_FILE,
    # )
    # args = parser.parse_args()

    main(SAMPLE_FILE)
