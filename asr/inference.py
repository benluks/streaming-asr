import torch
from .asr import load_asr_model, get_encoding

def create_inference_process(q, task, output_q=None):
    """
    Processes audio chunks from the queue and runs ASR or encoding.

    Args:
        q (mp.Queue): Queue containing audio chunks.
        mode (str): Either "asr" for transcription or "encode" for encoding.
    """
    asr_model, context = load_asr_model()
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

