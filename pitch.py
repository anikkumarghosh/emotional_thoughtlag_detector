# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_reaction_lag(file_path, energy_threshold=0.02):
#     # Load audio as numbers
#     y, sr = librosa.load("untitled.wav", sr=None)  # y = waveform array, sr = sample rate

#     print(f"Sample Rate: {sr} samples/sec")
#     print("First 50 samples (raw audio numbers):")
#     print(y[:50])  # show first few raw values

#     # Parameters
#     hop_length = 512  # step size between frames
#     frame_length = 1024  # samples in each frame

#     # Calculate RMS energy
#     rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

#     # Find first frame above threshold
#     frames = np.nonzero(rms > energy_threshold)[0]
#     if len(frames) == 0:
#         print("⚠ No speech detected.")
#         return None
#     first_frame = frames[0]

#     # Convert frame index to seconds
#     lag_sec = first_frame * hop_length / sr

#     # Plot waveform + RMS curve
#     times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)

#     plt.figure(figsize=(12, 6))
#     plt.plot(np.linspace(0, len(y)/sr, num=len(y)), y, alpha=0.5, label="Waveform")
#     plt.plot(times, rms, color='red', label="RMS Energy")
#     plt.axhline(y=energy_threshold, color='green', linestyle='--', label="Threshold")
#     plt.axvline(x=lag_sec, color='purple', linestyle='--', label=f"Speech Start: {lag_sec:.2f}s")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude / Energy")
#     plt.title("Reaction Lag Detection")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     return round(lag_sec, 3)

# if __name__ == "__main__":
#     audio_file = "sample.wav"  # replace with your file
#     lag = detect_reaction_lag(audio_file)
#     if lag is not None:
#         print(f"🕒 Reaction Lag: {lag} seconds")
######################################################1st implementation#######################################

# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_reaction_lag_and_pitch(file_path, energy_threshold=0.02):
#     # Load audio as numbers
#     y, sr = librosa.load("untitled.wav", sr=None)  # y = waveform array, sr = sample rate

#     print(f"Sample Rate: {sr} samples/sec")
#     print("First 50 samples (raw audio numbers):")
#     print(y[:50])  # show first few raw values

#     # Parameters
#     hop_length = 512  # step size between frames
#     frame_length = 1024  # samples in each frame

#     # === 1. Calculate RMS energy ===
#     rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

#     # Find first frame above threshold
#     frames = np.nonzero(rms > energy_threshold)[0]
#     if len(frames) == 0:
#         print("⚠ No speech detected.")
#         return None
#     first_frame = frames[0]
#     lag_sec = first_frame * hop_length / sr

#     # === 2. Pitch detection ===
#     f0, voiced_flag, voiced_probs = librosa.pyin(
#         y,
#         fmin=librosa.note_to_hz('C2'),  # low limit
#         fmax=librosa.note_to_hz('C7'),  # high limit
#         sr=sr,
#         hop_length=hop_length
#     )
#     mean_pitch = np.nanmean(f0)  # Average pitch ignoring NaNs
#     print(f"🎵 Average Pitch: {mean_pitch:.2f} Hz")

#     # === 3. Plot waveform + RMS + Pitch ===
#     times_rms = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
#     times_pitch = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=hop_length)

#     plt.figure(figsize=(14, 6))
#     plt.plot(np.linspace(0, len(y)/sr, num=len(y)), y, alpha=0.4, label="Waveform")
#     plt.plot(times_rms, rms, color='red', label="RMS Energy")
#     plt.plot(times_pitch, f0, color='blue', label="Pitch (Hz)")
#     plt.axhline(y=energy_threshold, color='green', linestyle='--', label="Threshold")
#     plt.axvline(x=lag_sec, color='purple', linestyle='--', label=f"Speech Start: {lag_sec:.2f}s")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Amplitude / Energy / Frequency")
#     plt.title("Reaction Lag + Pitch Detection")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     return {
#         "reaction_lag_sec": round(lag_sec, 3),
#         "average_pitch_hz": round(mean_pitch, 2)
#     }

# if __name__ == "__main__":
#     audio_file = "sample.wav"  # replace with your file
#     results = detect_reaction_lag_and_pitch(audio_file)
#     if results:
#         print(f"🕒 Reaction Lag: {results['reaction_lag_sec']} seconds")
#         print(f"🎵 Average Pitch: {results['average_pitch_hz']} Hz")

#############################################################################
# huggingface
#############################################################################
# Load model directly
# Requires: librosa
# from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
# import librosa
# import torch
# import numpy as np

# model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
# model = AutoModelForAudioClassification.from_pretrained(model_id)

# feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
# id2label = model.config.id2label
# def predict_emotion(audio_path, model, feature_extractor, id2label, max_duration=30.0):
#     inputs = preprocess_audio(audio_path, feature_extractor, max_duration)
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     inputs = {key: value.to(device) for key, value in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)

#     logits = outputs.logits
#     predicted_id = torch.argmax(logits, dim=-1).item()
#     predicted_label = id2label[predicted_id]
    
#     return predicted_label
# audio_path = "untitled.wav"

# predicted_emotion = predict_emotion(audio_path, model, feature_extractor, id2label)
# print(f"Predicted Emotion: {predicted_emotion}")
#####################################################################
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_pitch(file_path, fmin=50, fmax=500):
#     # Load audio
#     y, sr = librosa.load(file_path, sr=None)

#     # Extract pitch (fundamental frequency) with YIN
#     pitches = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)

#     # Time axis for plotting
#     times = librosa.times_like(pitches, sr=sr)

#     # Replace unvoiced regions (NaN) with 0
#     pitches_clean = np.where(np.isnan(pitches), 0, pitches)

#     # Plot pitch contour
#     plt.figure(figsize=(12, 6))
#     plt.plot(times, pitches_clean, label="Pitch (Hz)", color="blue")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Frequency (Hz)")
#     plt.title("Pitch Contour of Speech")
#     plt.legend()
#     plt.show()

#     # Basic stats
#     voiced = pitches_clean[pitches_clean > 0]
#     if len(voiced) > 0:
#         print(f"📊 Avg Pitch: {np.mean(voiced):.2f} Hz")
#         print(f"📈 Max Pitch: {np.max(voiced):.2f} Hz")
#         print(f"📉 Min Pitch: {np.min(voiced):.2f} Hz")
#     else:
#         print("⚠ No voiced pitch detected.")

#     return pitches_clean, times

# # --------------------------
# # Run pitch detection
# # --------------------------
# if __name__ == "__main__":
#     audio_file = "untitleed.wav"  # replace with your file
#     detect_pitch(audio_file)
################################################
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt

# def detect_pitch(file_path, fmin=50, fmax=500):
#     """
#     Detects pitch (fundamental frequency) from audio using YIN
#     Returns pitch contour, average pitch, and pitch deviation.
#     """
#     # Load audio
#     y, sr = librosa.load(file_path, sr=None)

#     # Extract pitch using YIN (better for monophonic signals like voice)
#     pitches = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)

#     # Time axis
#     times = librosa.times_like(pitches, sr=sr)

#     # Remove NaN (unvoiced frames)
#     pitches_clean = np.where(np.isnan(pitches), 0, pitches)

#     # Keep only voiced regions
#     voiced = pitches_clean[pitches_clean > 0]

#     # Plot pitch contour
#     plt.figure(figsize=(12, 6))
#     plt.plot(times, pitches_clean, label="Pitch Contour (Hz)", color="blue", alpha=0.8)
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Frequency (Hz)")
#     plt.title("Pitch Contour of Speech")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

#     # Stats
#     if len(voiced) > 0:
#         avg_pitch = np.mean(voiced)
#         std_pitch = np.std(voiced)  # deviation = pitch variability
#         min_pitch = np.min(voiced)
#         max_pitch = np.max(voiced)

#         print("📊 Pitch Statistics")
#         print(f"   Avg Pitch: {avg_pitch:.2f} Hz")
#         print(f"   Pitch Deviation (Std): {std_pitch:.2f} Hz")
#         print(f"   Min Pitch: {min_pitch:.2f} Hz")
#         print(f"   Max Pitch: {max_pitch:.2f} Hz")
#     else:
#         print("⚠ No voiced pitch detected.")
#         avg_pitch, std_pitch, min_pitch, max_pitch = None, None, None, None

#     return pitches_clean, times, avg_pitch, std_pitch

# # --------------------------
# # Run pitch detection
# # --------------------------
# if __name__ == "__main__":
#     audio_file = "untitleed.wav"  # replace with your file
#     detect_pitch(audio_file)
####################################################################
import librosa
import numpy as np
import matplotlib.pyplot as plt

def detect_pitch(file_path, fmin=50, fmax=500):
    """
    Detects pitch (fundamental frequency) from audio using YIN
    Returns pitch contour, average pitch, and pitch deviation.
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Extract pitch using YIN (better for monophonic signals like voice)
    pitches = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)

    # Time axis
    times = librosa.times_like(pitches, sr=sr)

    # Remove NaN (unvoiced frames)
    pitches_clean = np.where(np.isnan(pitches), 0, pitches)

    # Keep only voiced regions
    voiced = pitches_clean[pitches_clean > 0]

    # Plot pitch contour
    plt.figure(figsize=(12, 6))
    plt.plot(times, pitches_clean, label="Pitch Contour (Hz)", color="blue", alpha=0.8)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour of Speech")

    # Stats
    if len(voiced) > 0:
        avg_pitch = np.mean(voiced)
        std_pitch = np.std(voiced)  # deviation = pitch variability
        min_pitch = np.min(voiced)
        max_pitch = np.max(voiced)

        # Add text box with stats
        stats_text = (
            f"Avg Pitch: {avg_pitch:.2f} Hz\n"
            f"Pitch Deviation: {std_pitch:.2f} Hz\n"
            f"Min Pitch: {min_pitch:.2f} Hz\n"
            f"Max Pitch: {max_pitch:.2f} Hz"
        )
        plt.gca().text(
            0.98, 0.95, stats_text,
            transform=plt.gca().transAxes,
            fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.7)
        )

        print("📊 Pitch Statistics")
        print(stats_text)
    else:
        print("⚠ No voiced pitch detected.")
        avg_pitch, std_pitch, min_pitch, max_pitch = None, None, None, None

    plt.legend()
    plt.tight_layout()
    plt.show()

    return pitches_clean, times, avg_pitch, std_pitch

# --------------------------
# Run pitch detection
# --------------------------
if __name__ == "__main__":
    audio_file = "untitled.wav"  # replace with your file
    detect_pitch(audio_file)
