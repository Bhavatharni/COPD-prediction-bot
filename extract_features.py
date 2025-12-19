# extract_features.py
import os
import glob
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = r"F:\Major Project\project 1\dataset\Respiratory_Sound_Database"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_and_txt_files")
DIAG_PATH = os.path.join(DATA_DIR, "patient_diagnosis.csv")  # optional
OUT_CSV = os.path.join(DATA_DIR, "finalalldata.csv")

def extract_features_from_file(path, sr=22050, n_mfcc=13):
    y, sr = librosa.load(path, sr=sr, mono=True)
    # Basic features
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    # Spectral
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean, zcr_std = zcr.mean(), zcr.std()
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    sc_mean, sc_std = spec_cent.mean(), spec_cent.std()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sb_mean, sb_std = spec_bw.mean(), spec_bw.std()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    ro_mean, ro_std = rolloff.mean(), rolloff.std()
    rms = librosa.feature.rms(y=y)
    rms_mean, rms_std = rms.mean(), rms.std()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    # Collect features in dict
    feats = {}
    # MFCC mean & std
    for i in range(n_mfcc):
        feats[f"mfcc_mean_{i+1}"] = float(mfcc_mean[i])
        feats[f"mfcc_std_{i+1}"] = float(mfcc_std[i])
    feats["zcr_mean"] = float(zcr_mean)
    feats["zcr_std"] = float(zcr_std)
    feats["spec_cent_mean"] = float(sc_mean)
    feats["spec_cent_std"] = float(sc_std)
    feats["spec_bw_mean"] = float(sb_mean)
    feats["spec_bw_std"] = float(sb_std)
    feats["rolloff_mean"] = float(ro_mean)
    feats["rolloff_std"] = float(ro_std)
    feats["rms_mean"] = float(rms_mean)
    feats["rms_std"] = float(rms_std)
    # Chroma
    for i in range(chroma_mean.shape[0]):
        feats[f"chroma_mean_{i+1}"] = float(chroma_mean[i])
        feats[f"chroma_std_{i+1}"] = float(chroma_std[i])
    # duration & sampling
    feats["duration"] = float(librosa.get_duration(y=y, sr=sr))
    feats["sr"] = int(sr)
    return feats

def patient_id_from_filename(fname):
    # expected filename like "101_1b1_Al_sc_Meditron.wav" -> patient id 101
    base = os.path.basename(fname)
    parts = base.split("_")
    try:
        pid = int(parts[0])
    except:
        pid = None
    return pid

def main():
    wavs = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
    rows = []
    print(f"Found {len(wavs)} wav files.")
    for p in tqdm(wavs):
        try:
            feats = extract_features_from_file(p)
            base = os.path.basename(p)
            feats["filename"] = base
            feats["patient_id"] = patient_id_from_filename(base)
            rows.append(feats)
        except Exception as e:
            print(f"Failed on {p}: {e}")
    df = pd.DataFrame(rows)
    # If diagnosis CSV present, merge
    if os.path.exists(DIAG_PATH):
        diag = pd.read_csv(DIAG_PATH)
        # normalize column names
        if "patient_id" not in diag.columns:
            # try to infer columns
            cols = diag.columns.tolist()
            if len(cols) >= 2:
                diag = diag.rename(columns={cols[0]:"patient_id", cols[1]:"diagnosis"})
        # map diagnosis to binary label
        def map_label(x):
            if isinstance(x, str) and x.strip().lower() == "copd":
                return 1
            # some rows might contain multiple conditions separated by ';' or ','
            if isinstance(x, str) and "copd" in x.lower():
                return 1
            return 0
        diag["label"] = diag["diagnosis"].apply(map_label)
        # ensure patient_id types match
        diag["patient_id"] = pd.to_numeric(diag["patient_id"], errors="coerce")
        df = df.merge(diag[["patient_id", "label"]], on="patient_id", how="left")
        # some files may not have labels -> fill -1 or 0; we choose to drop unlabeled rows
        missing_label = df["label"].isna().sum()
        print(f"After merging diagnosis, missing labels: {missing_label}")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    else:
        print("No patient_diagnosis.csv found. finalalldata.csv will be feature-only (no label).")
    # Save CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved features to {OUT_CSV} (rows: {len(df)})")

if __name__ == "__main__":
    main()
