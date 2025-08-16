import librosa
import numpy as np
import warnings

# Your code that might generate warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


def detect_reaction_lag(file_path, energy_threshold=0.02):
    y, sr = librosa.load("untitled.wav", sr=None)  # y = waveform array, sr = sample rate

    print(f"Sample Rate: {sr} samples/sec")
    print("First 50 samples (raw audio numbers):")
    print(y[:50])  # show first few raw values

    hop_length = 512  # step size between frames
    frame_length = 1024  # samples in each frame

    # Calculate RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # Find first frame above threshold
    frames = np.nonzero(rms > energy_threshold)[0]
    if len(frames) == 0:
        print("⚠ No speech detected.")
        return None
    first_frame = frames[0]

    # Convert frame index to seconds
    lag_sec = first_frame * hop_length / sr

    # Plot waveform + RMS curve
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

    plt.figure(figsize=(12, 6))
    plt.plot(np.linspace(0, len(y)/sr, num=len(y)), y, alpha=0.5, label="Waveform")
    plt.plot(times, rms, color='red', label="RMS Energy")
    plt.axhline(y=energy_threshold, color='green', linestyle='--', label="Threshold")
    plt.axvline(x=lag_sec, color='purple', linestyle='--', label=f"Speech Start: {lag_sec:.2f}s")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude / Energy")
    plt.title("Reaction Lag Detection")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return round(lag_sec, 3)

if __name__ == "__main__":
    audio_file = "untitleed.wav"  # replace with your file
    lag = detect_reaction_lag(audio_file)
    if lag is not None:
        print(f"🕒 Reaction Lag: {lag} seconds")