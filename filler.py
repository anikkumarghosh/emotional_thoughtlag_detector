import whisper
import re
from collections import Counter


print("Loading Whisper model...")
model = whisper.load_model("medium")  # You can use "small" for faster but less accurate
print("Model loaded.\n")


audio_file = "untitleed.wav"  # Replace with your audio file path
print(f"Transcribing {audio_file} ...")
result = model.transcribe(audio_file, word_timestamps=True)
transcript = result['text']
print("Transcription complete.\n")


# ------
filler_words = ["um", "uh", "like", "you know", "so", "actually", "basically"]
transcript_lower = transcript.lower()

filler_count = {
    word: len(re.findall(r'\b' + re.escape(word) + r'\b', transcript_lower))
    for word in filler_words
}


# Matches patterns like "I-I-I", "so-so-so", etc.
repeats = re.findall(r'\b(\w+)(-\1)+\b', transcript)
repeat_count = len(repeats)


total_words = len(transcript.split())
total_fillers = sum(filler_count.values())

print("\n--- Transcript ---")
print(transcript)

print("\n--- Speech Analysis ---")
print(f"Total Words Spoken: {total_words}")
print(f"Total Filler Words: {total_fillers}")
print(f"Percentage of Filler Words: {total_fillers / total_words * 100:.2f}%")
print(f"Repeated Words / Stutters Count: {repeat_count}")
print(f"Filler Word Breakdown: {filler_count}")

# Optional: Highlight filler words in transcript
highlighted_transcript = transcript
for word in filler_words:
    highlighted_transcript = re.sub(r'\b' + re.escape(word) + r'\b', f"[{word.upper()}]", highlighted_transcript, flags=re.IGNORECASE)

print("\n--- Transcript with Filler Words Highlighted ---")
print(highlighted_transcript)
