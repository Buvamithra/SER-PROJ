# 🎤 Speech Emotion Recognition (SER) using Deep Learning

This project is a **Speech Emotion Recognition (SER)** system that detects human emotions from speech audio using a Deep Neural Network. It classifies emotions such as **Angry**, **Happy**, **Sad**, **Fearful**, **Calm**, **Disgust**, and **Neutral** by analyzing audio features (MFCCs) and can even respond to high-risk emotional states in real-time.

> 🎯 Ideal for applications in mental health monitoring, customer service analytics, and interactive AI agents.

---

## 📌 Features

- 🎙️ Real-time emotion recognition from live microphone input
- 📁 Trained on **RAVDESS** — a well-known emotional speech dataset
- 📊 MFCC feature extraction and visualization
- 🧠 DNN-based model achieving high accuracy
- 🔔 Alerts when high-risk emotions (like *Angry*, *Fear*, or *Anxiety*) are detected with >98% confidence
- 💻 Simple GUI for interaction

---

## 🧠 Emotion Classes

The model predicts the following emotions:

- 😠 Angry  
- 😢 Sad  
- 😨 Fear  
- 😄 Happy  
- 😐 Neutral  
- 😌 Calm  
- 🤢 Disgust

---

## 📂 Folder Structure

SER-Project/
├── dataset/ # RAVDESS dataset (download externally)
├── extracted_features/ # CSV files containing MFCC features
├── mfcc_plots/ # MFCC plots for visual inspection
├── models/ # Saved trained model (.h5)
├── real_time_test.py # Real-time microphone testing
├── train_model.py # Training script for the model
├── feature_extraction.py # Extract MFCCs from dataset
├── final_gui.py # GUI interface for the system
├── alarm.mp3 # Alert for high-risk emotions
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/speech-emotion-recognition.git
cd speech-emotion-recognition
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Download the RAVDESS Dataset
Download it from Zenodo and place it inside the dataset/ folder.

⚙️ Run the Project
➤ Extract MFCC Features
bash
Copy
Edit
python feature_extraction.py
This will save features in extracted_features/ and MFCC plots in mfcc_plots/.

➤ Train the Model
bash
Copy
Edit
python train_model.py
Trained model will be saved in the models/ directory.

➤ Real-time Emotion Recognition
bash
Copy
Edit
python real_time_test.py
This will record 10 seconds from your mic, detect emotion, and play an alert if needed.

➤ GUI Interface (Optional)
bash
Copy
Edit
python final_gui.py
Includes emotion prediction display, background visuals, and alert sounds.

🔊 High-Risk Emotion Alert
If an emotion such as Angry, Fear, or Anxiety is detected with more than 98% confidence, the system plays an alert sound (alarm.mp3). This feature is useful for:

🚨 Emergency response systems

🧓 Elderly/child emotion tracking

🧠 Mental wellness monitoring

📈 Model Details
Algorithm: Deep Neural Network (Keras, TensorFlow backend)

Input Features: 40 MFCCs per sample

Dataset: RAVDESS (Audio-only subset)

Validation Accuracy: ~95%+

Loss Function: Categorical Crossentropy

Optimizer: Adam

📊 Visual Output
MFCC plots for each audio file

Accuracy/Loss curve plots

Confusion matrix for final evaluation

🖼️ Sample Screenshot
(Add a screenshot here if you'd like. You can use a GUI screenshot or MFCC plots.)

🧪 Requirements
All dependencies are listed in requirements.txt. Install them using:

bash
Copy
Edit
pip install -r requirements.txt
Main libraries used:

librosa

tensorflow / keras

numpy, pandas

sounddevice

matplotlib

scikit-learn

tkinter (for GUI)

📬 Author
Buvamithra Ulagarajan
🎓 Final Year B.E (ECE), Aspiring AI Engineer
📧 buvamithraulagarajan@gmail.com
🔗 LinkedIn Profile

📄 License
This project is open-source and available under the MIT License. Feel free to modify, distribute, and use for academic or commercial purposes.
