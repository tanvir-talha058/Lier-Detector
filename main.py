import cv2
from deepface import DeepFace
import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import os
from tkinter import ttk  # Import themed tkinter


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

# Parameters for voice recording
DURATION = 5  # seconds
FILENAME = "voice_recording.wav"
SAMPLE_RATE = 44100

# --- Helper Functions ---
def analyze_frame(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        dominant_emotion = analysis[0]['dominant_emotion']
        score = truth_likelihood.get(dominant_emotion, 0.5)
        print(f"Detected Emotion: {dominant_emotion}")
        print(f"Estimated Truth Likelihood: {score * 100:.2f}%")
        return dominant_emotion, score  # Return for display
    except Exception as e:
        print(f"Error: {e}")
        return "Error", 0.5

def capture_and_analyze():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video device")
        return

    print("Press SPACE to capture image and analyze, or ESC to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Check camera connection.")
            break

        cv2.imshow('Truth Detector - Press SPACE to analyze', frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            emotion, score = analyze_frame(frame)
            messagebox.showinfo("Face Analysis", f"Detected Emotion: {emotion}\nEstimated Truth Likelihood: {score * 100:.2f}%")
            break  # Analyze only once per button press

    cap.release()
    cv2.destroyAllWindows()

def analyze_voice(file_path):
    try:
        y, sr = librosa.load(file_path)
        pitch, _ = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitch[pitch > 0])
        energy = np.mean(librosa.feature.rms(y=y))

        if pitch_mean < 120 and energy < 0.01:
            emotion = "nervous"
            truth_score = 0.3
        elif 120 <= pitch_mean <= 180:
            emotion = "neutral/calm"
            truth_score = 0.8
        elif pitch_mean > 180:
            emotion = "excited/anxious"
            truth_score = 0.5
        else:
            emotion = "unknown"
            truth_score = 0.5

        return emotion, truth_score
    except Exception as e:
        print(f"Error analyzing voice: {e}")
        return "unknown", 0.5

def record_and_analyze():
    try:
        messagebox.showinfo("Recording", f"Recording for {DURATION} seconds...")
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        write(FILENAME, SAMPLE_RATE, audio)
        messagebox.showinfo("Recording", "Recording finished!")

        emotion, truth_score = analyze_voice(FILENAME)
        messagebox.showinfo("Voice Analysis", f"Detected Emotion: {emotion}\nEstimated Truth Likelihood: {truth_score * 100:.2f}%")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during voice recording/analysis: {e}")

# --- GUI Setup ---
def create_gui():
    window = tk.Tk()
    window.title("Truth Detector")
    window.geometry("400x300")  # Set a reasonable window size
    window.configure(bg="#f0f0f0")  # Light background

    # Style configuration (using ttk)
    style = ttk.Style()
    style.configure("TButton",
                    padding=10,
                    font=('Helvetica', 12),
                    background="#4CAF50",  # Green
                    foreground="white")
    style.map("TButton",
              background=[("active", "#388E3C")],  # Darker green on hover
              foreground=[("active", "white")])

    style.configure("TLabel",
                    background="#f0f0f0",
                    font=('Helvetica', 14))

    # Title Label
    title_label = ttk.Label(window, text="Truth Detector", font=('Helvetica', 18, 'bold'))
    title_label.pack(pady=20)

    # Buttons
    face_button = ttk.Button(window, text="Analyze Face", command=capture_and_analyze)
    face_button.pack(pady=10)

    voice_button = ttk.Button(window, text="Analyze Voice", command=record_and_analyze)
    voice_button.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    create_gui()
