import torch
import torchaudio.io as taio
import multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neural network for audio processing
class AudioProcessingNet(nn.Module):
    def __init__(self):
        super(AudioProcessingNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.fc = nn.Linear(3997, 1)  # Example output size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x.squeeze(-1)

# Function to capture audio from the microphone
def audio_capture_process(queue):
    # Initialize the StreamReader with the default microphone
    streamer = taio.StreamReader(src=":3", format="avfoundation")  # Default audio device
    streamer.add_basic_audio_stream(frames_per_chunk=16000, sample_rate=16000, num_channels=1)  # 1-second chunks at 16kHz

    # Start streaming
    for (audio,) in streamer.stream(-1):
        # Send audio data to the processing process
        print(audio.shape)
        queue.put(audio)

# Function to process audio using the neural network
def audio_processing_process(queue):
    # Initialize the neural network
    model = AudioProcessingNet()
    model.eval()  # Set the model to evaluation mode

    # Process audio frames from the queue
    while True:
        audio = queue.get()  # Get audio data from the queue
        print(f"Audio got: {audio.shape}")
        if audio is None:  # Sentinel value to stop the process
            break

        # Preprocess the audio (e.g., normalize, add batch dimension)
        audio = audio.squeeze(-1).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(audio)
            print("Network output:", output)

if __name__ == "__main__":
    # Explicitly create a spawn context
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()

    # Start the audio capture process
    capture_process = ctx.Process(target=audio_capture_process, args=(queue,))
    capture_process.start()

    # Start the audio processing process
    processing_process = ctx.Process(target=audio_processing_process, args=(queue,))
    processing_process.start()

    # Wait for the processes to finish (you can add a termination condition)
    capture_process.join()
    processing_process.join()