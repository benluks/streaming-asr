# **Real-Time Speech Recognition with SpeechBrain**

This project enables **real-time automatic speech recognition (ASR)** using **SpeechBrain's [StreamingASR](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.ASR.html#speechbrain.inference.ASR.StreamingASR)** and **TorchAudio's [StreamReader](https://pytorch.org/audio/main/tutorials/streamreader_basic_tutorial.html)**. It supports both **live microphone streaming** and **file-based transcription**.

## **Features**

- **Live ASR from Microphone**
- **Transcription from an Audio File or URL**
- **Streaming-based Processing for Low Latency**
- **Easily Configurable Input Sources**

---

## **Dependencies**

You'll need **Python 3.8+**, `torch`, `torchaudio`, and `speechbrain`:

```sh
pip install torch torchaudio speechbrain
```

## **Usage**

### Microphone Transcription (Default)

```sh
python main.py
```

This captures audio from your default microphone and transcribes it in real-time.
You can modify `DEVICE` and `SRC` in main.py to match your system. Check [here](git rm --cached <file_or_folder>
) to learn about `StreamReader` configurations.

### File-Based Transcription

```sh
python main.py --file path/to/audio.wav
```

or, from a **URL**:

```sh
python main.py -f https://upload.wikimedia.org/wikipedia/commons/transcoded/9/97/Spoken_Wikipedia_-_One_Times_Square.ogg/Spoken_Wikipedia_-_One_Times_Square.ogg.mp3
```

## Notes

### Finding Available Audio Devices

On macOS, list available input devices using:

```sh
ffmpeg -f avfoundation -list_devices true -i dummy
```

On Linux:

```sh
arecord -l
```

Modify `SRC` and `DEVICE` in main.py accordingly.

### `ffmpeg` requirements

`torchaudio` has particular requirements for ffmpeg. You can read about that [here](https://pytorch.org/audio/main/installation.html#ffmpeg-dependency).

## Batch Encoding Workflow

This script performs batch encoding of pre-recorded vowel audio files using a pretrained ASR model. It supports saving encodings in two formats:

- **Encoding dictionary (.pt)**: For each vowel, stores the model's output and optionally the hidden states.
- **Full tensor format (.pkl)**: A single tensor with all vowel encodings concatenated along the time axis, along with label and length metadata.

Usage

1. Encode to dictionary format:

```bash
python -m tasks.encode_batch \
 --output vowel_encodings/encodings_hidden_states_dict.pt \
 --hidden_states \
 --only_keep_vowels
```

This saves a .pt file where each vowel maps to:

```
{
"output": Tensor, # [D, T]
"hidden_states": [Tensor, Tensor, ...] # optional
}
```

2. Convert encoding dictionary to full tensor:

```bash
python -m tasks.encode_batch \
 --full_encoding \
 --encoding_dict_path vowel_encodings/encodings_hidden_states_dict.pt \
 --output vowel_encodings/encodings_hidden_states.pkl
```

This creates a .pkl file with:

```
{
"encodings": ndarray of shape [B, D, T],
"labels": List[str],
"lengths": List[int]
}
```

### Notes

- If `--full_encoding` is used without specifying a different `--output`, the `.pt` file may be overwritten.
- `--relu` applies ReLU to the encodings and hidden states before saving.
- `--only_keep_vowels` filters to a predefined set of common vowels (`KEEP_VOWELS`).
