# chatbot_api.py  (with audio prediction support)

import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
import librosa
import torch
import torch.nn as nn

DATA_DIR = r"F:\Major Project\project 1\dataset\Respiratory_Sound_Database"
MODEL_DIR = os.path.join(DATA_DIR, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
META_PATH = os.path.join(MODEL_DIR, "meta.joblib")
CLASS_STATE = os.path.join(MODEL_DIR, "classifier_state.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- MODEL CLASS ----
class DBNClassifier(nn.Module):
    def __init__(self, layer_sizes, n_classes=2):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ---- FLASK APP ----
app = Flask(__name__)

GLOBAL = {"model": None, "scaler_info": None, "cols": None}

# -------- LOAD MODEL --------
def load_model():
    if GLOBAL["model"] is not None:
        return

    scaler_info = joblib.load(SCALER_PATH)
    meta = joblib.load(META_PATH)
    columns = scaler_info["columns"]

    input_dim = meta["input_dim"]
    hidden_sizes = meta["rbm_sizes"]
    layer_sizes = [input_dim] + hidden_sizes

    model = DBNClassifier(layer_sizes)
    model.load_state_dict(torch.load(CLASS_STATE, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    GLOBAL["model"] = model
    GLOBAL["scaler_info"] = scaler_info
    GLOBAL["cols"] = columns


# -------- AUDIO FEATURE EXTRACTION --------
def extract_features_from_audio(audio_path, sr=22050, n_mfcc=13):

    y, sr = librosa.load(audio_path, sr=sr)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = float(np.mean(zcr))
    zcr_std = float(np.std(zcr))

    # Spectral
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean = float(np.mean(spec_cent))
    sc_std = float(np.std(spec_cent))

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Build dict
    features = {}

    # MFCCs
    for i in range(n_mfcc):
        features[f"mfcc_mean_{i+1}"] = float(mfcc_mean[i])
        features[f"mfcc_std_{i+1}"] = float(mfcc_std[i])

    # ZCR + spectral
    features["zcr_mean"] = zcr_mean
    features["zcr_std"] = zcr_std
    features["spec_cent_mean"] = sc_mean
    features["spec_cent_std"] = sc_std
    features["duration"] = duration

    return features


# -------- PREPROCESS FEATURES --------
def preprocess(features_dict):
    info = GLOBAL["scaler_info"]
    columns = GLOBAL["cols"]

    arr = []
    for col in columns:
        arr.append(features_dict.get(col, np.nan))

    arr = np.array(arr).reshape(1, -1)
    arr = info["imputer"].transform(arr)
    arr = info["scaler"].transform(arr)
    return arr


# -------- POST /predict_audio --------
@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    load_model()

    if "file" not in request.files:
        return jsonify({"error": "Upload a WAV file using form-data"}), 400

    file = request.files["file"]

    # Save temporary file
    temp_path = "temp.wav"
    file.save(temp_path)

    # Extract features
    feats = extract_features_from_audio(temp_path)
    x = preprocess(feats)

    # Predict
    model = GLOBAL["model"]
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=DEVICE))
        prob = float(torch.softmax(logits, dim=1)[0][1].cpu())
        pred = int(prob >= 0.5)

    return jsonify({
        "prediction": pred,
        "probability_of_copd": prob
    })


# -------- MAIN --------
if __name__ == "__main__":
    print("API running: http://127.0.0.1:5000")
    app.run(debug=True)
