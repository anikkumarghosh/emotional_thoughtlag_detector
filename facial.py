from transformers import pipeline
import cv2
from PIL import Image


def main():
    """Run live emotion detection from the default webcam."""

    # Example with a specific model
    emotion_pipeline = pipeline("image-classification", model="trpakov/vit-face-expression")

    cap = cv2.VideoCapture(0)  # 0 for default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert OpenCV BGR image to PIL RGB image
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Perform emotion inference
            predictions = emotion_pipeline(pil_image)

            # Extract dominant emotion and display on frame
            if predictions:
                dominant_emotion = predictions[0]["label"]
                score = predictions[0]["score"]
                text = f"{dominant_emotion}: {score:.2f}"
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Live Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()