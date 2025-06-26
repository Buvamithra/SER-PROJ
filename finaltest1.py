import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
import librosa
import joblib
import noisereduce as nr
from sklearn.preprocessing import StandardScaler
from playsound import playsound

# === Load your trained model ===
model = joblib.load("ser_model.pkl")

# === Set high-risk emotion class names ===
high_risk_emotions = ['Angry', 'Fear', 'Sad']

# === Emotion labels (update if needed) ===
emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']

# === GUI Setup ===
app = tk.Tk()
app.title("SPEECH EMOTION DETECTION")
app.geometry("500x300")
app.configure(bg='black')

label = Label(app, text="Drag and drop or choose a .wav file", fg='white', bg='black', font=('Helvetica', 14))
label.pack(pady=20)

result_label = Label(app, text="", font=('Helvetica', 18, 'bold'), bg='black', fg='lime')
result_label.pack(pady=20)

# === Function to process and predict emotion ===
def predict_emotion(filepath):
    try:
        y, sr = librosa.load(filepath, sr=22050)
        y_denoised = nr.reduce_noise(y=y, sr=sr)

        mfcc = librosa.feature.mfcc(y=y_denoised, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)

        scaler = StandardScaler()
        mfcc_scaled = scaler.fit_transform(mfcc_scaled)

        prediction = model.predict(mfcc_scaled)[0]
        emotion = prediction if isinstance(prediction, str) else emotion_labels[int(prediction)]

        result_label.config(text=emotion)

        if emotion in high_risk_emotions:
            playsound('alarm.wav')  # You must have alarm.mp3 in the same folder

    except Exception as e:
        result_label.config(text=f"Error: {str(e)}")

# === Browse and Load File ===
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        label.config(text=os.path.basename(file_path))
        predict_emotion(file_path)

# === Button to browse ===
btn = Button(app, text="Predict", command=browse_file, font=('Helvetica', 14, 'bold'), bg='red', fg='white')
btn.pack(pady=10)

# === Run the GUI ===
app.mainloop()
