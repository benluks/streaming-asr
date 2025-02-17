# **Real-Time Speech Recognition with SpeechBrain**

This project enables **real-time automatic speech recognition (ASR)** using **SpeechBrain** and **TorchAudio's StreamReader**. It supports both **live microphone streaming** and **file-based transcription**.

## **🚀 Features**
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
You can modify `DEVICE` and `SRC` in main.py to match your system. Check [here] for `StreamReader` configurations.

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