import torch
import librosa
import numpy as np
import joblib

from models.audio_model import AudioEmotionModel


class AudioInference:

    def __init__(self):
        # load model
        self.model = AudioEmotionModel(input_dim=40, num_classes=8)
        self.model.load_state_dict(torch.load("models/audio_emotion_model.pth"))
        self.model.eval()

        # load preprocessing objects
        self.scaler = joblib.load("models/scaler.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")

    def extract_features(self, file_path):
        audio, sr = librosa.load(file_path, duration=3, offset=0.5)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    def predict(self, file_path):

        features = self.extract_features(file_path)

        features = self.scaler.transform([features])

        tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            outputs = self.model(tensor)

            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, dim=1)

        emotion = self.label_encoder.inverse_transform(
            [predicted.item()]
        )[0]

        return emotion, confidence.item()