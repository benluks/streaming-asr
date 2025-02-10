def stream(queue):
    from torchaudio.io import StreamReader

    streamer = StreamReader(
        src=":3",
        format="avfoundation",
    )
    streamer.add_basic_audio_stream(frames_per_chunk=7056, sample_rate=16000)

    while True:
        stream_iterator = streamer.stream(-1)
        (chunk,) = next(stream_iterator)
        queue.put(chunk)


import torch.multiprocessing as mp

ctx = mp.get_context("spawn")
queue = ctx.Queue()
p = ctx.Process(target=stream, args=(queue,))
p.start()
