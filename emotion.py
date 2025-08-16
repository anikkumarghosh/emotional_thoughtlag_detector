import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


MODEL_PATH = "./final_model" 
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)

def predict_emotion(audio_path):
    
    # Load the audio file. The model requires a 16kHz sample rate.
    speech_array, sample_rate = librosa.load(audio_path, sr=16000)

    inputs = feature_extractor(
        speech_array, 
        sampling_rate=16000, 
        padding=True, 
        truncation=True,
        max_length=16000 * 5, # Max 5 seconds
        return_tensors="pt"  # Return PyTorch tensors
    )

    # Make the prediction
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Get the class with the highest probability
    predicted_id = torch.argmax(logits, dim=-1).item()
    
    # Convert the ID to its human-readable label
    predicted_emotion = model.config.id2label[predicted_id]
    
    return predicted_emotion

# --- Main execution block ---
if __name__ == "__main__":

    new_audio_file = "untitled.wav" 

    # --- 3. Get the prediction ---
    try:
        emotion = predict_emotion(new_audio_file)
        print(f"✅ The predicted emotion for the file is: {emotion}")
    except FileNotFoundError:
        print(f"❌ Error: The file '{new_audio_file}' was not found.")
        print("Please make sure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")