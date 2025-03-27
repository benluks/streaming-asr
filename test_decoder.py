import logging
logging.getLogger("speechbrain").setLevel(logging.WARNING)

import torch
from torchaudio.io import StreamReader
from asr import load_asr_model, transcribe
from decoder import build_decoder

CHUNK_SIZE = 8
CHUNK_SAMPLES = 639
SAMPLE_FILE = "https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3"

streamer = StreamReader(src=SAMPLE_FILE)
asr_model, context = load_asr_model(CHUNK_SIZE, 2)
decoder = build_decoder()

sr = asr_model.hparams.sr
frames_per_chunk = CHUNK_SIZE * CHUNK_SAMPLES

streamer.add_basic_audio_stream(
    frames_per_chunk=frames_per_chunk, sample_rate=sr, num_channels=1
)
logits_buffer = torch.tensor()

for (chunk,) in streamer.stream(-1):
    
    with torch.no_grad():
        chunk = chunk.squeeze(-1).unsqueeze(0).float()
        # factory_words = transcribe_chunk(asr_model, context, chunk)
        logits, words = transcribe(asr_model, context, chunk)
        logits_buffer = torch.cat((logits_buffer, logits))
    

