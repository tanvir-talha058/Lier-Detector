import cv2
from deepface import DeepFace

# Scoring system (for demonstration only)
truth_likelihood = {
    "happy": 0.9,
    "neutral": 0.8,
    "surprise": 0.7,
    "sad": 0.4,
    "fear": 0.3,
    "angry": 0.2,
    "disgust": 0.1
}

def analyze_frame(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        score = truth_likelihood.get(dominant_emotion, 0.5)
        print(f"Detected Emotion: {dominant_emotion}")
        print(f"Estimated Truth Likelihood: {score * 100:.2f}%")
    except Exception as e:
        print(f"Error: {e}")

def capture_and_analyze():
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture image and analyze, or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Truth Detector - Press SPACE to analyze', frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            analyze_frame(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_analyze()
