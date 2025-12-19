# train_dbn.py
import os
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

import torch
import torch.nn as nn
import torch.optim as optim

DATA_DIR = r"F:\Major Project\project 1\dataset\Respiratory_Sound_Database"
CSV_PATH = os.path.join(DATA_DIR, "finalalldata.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
RBMS_STATE = os.path.join(MODEL_DIR, "rbms_state.pth")
CLASS_STATE = os.path.join(MODEL_DIR, "classifier_state.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ---- RBM implementation (same as earlier)
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid, k=1):
        super().__init__()
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.k = k
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.v_bias = nn.Parameter(torch.zeros(n_vis))

    def sample_h(self, v):
        pre = torch.matmul(v, self.W.t()) + self.h_bias
        p_h = torch.sigmoid(pre)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h):
        pre = torch.matmul(h, self.W) + self.v_bias
        p_v = torch.sigmoid(pre)
        return p_v, torch.bernoulli(p_v)

    def free_energy(self, v):
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = torch.matmul(v, self.W.t()) + self.h_bias
        hidden_term = torch.sum(torch.log1p(torch.exp(wx_b)), dim=1)
        return -vbias_term - hidden_term

    def contrastive_divergence(self, v0, lr=0.01):
        v = v0
        p_h0, h0 = self.sample_h(v)
        for _ in range(self.k):
            p_v, v = self.sample_v(h0)
            p_h, h = self.sample_h(v)
            h0 = h
        pos = torch.matmul(p_h0.t(), v0)
        neg = torch.matmul(p_h.t(), v)
        batch_size = v0.size(0)
        dW = (pos - neg) / batch_size
        dvb = torch.mean(v0 - v, dim=0)
        dhb = torch.mean(p_h0 - p_h, dim=0)
        self.W.data += lr * dW
        self.v_bias.data += lr * dvb
        self.h_bias.data += lr * dhb
        loss = torch.mean(self.free_energy(v0)) - torch.mean(self.free_energy(v))
        return loss.item()

    def transform(self, X):
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            p_h = torch.sigmoid(torch.matmul(X_t, self.W.t()) + self.h_bias)
            return p_h.cpu().numpy()

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

# ---- Preprocessing & training
def load_data(path):
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("finalalldata.csv must contain a 'label' column (0/1)")
    X = df.drop(columns=["filename", "patient_id", "label"], errors="ignore")
    y = df["label"].values
    # impute & scale
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    joblib.dump({"imputer": imputer, "scaler": scaler, "columns": X.columns.tolist()}, SCALER_PATH)
    return X_scaled, y

def pretrain_rbms(X_train, rbm_sizes=[128,64], epochs=10, lr=0.01, batch_size=64):
    rbms = []
    data = X_train.copy()
    for i, hid in enumerate(rbm_sizes):
        n_vis = data.shape[1]
        print(f"Pretraining RBM {i+1}: {n_vis} -> {hid}")
        rbm = RBM(n_vis, hid).to(DEVICE)
        n_batches = math.ceil(len(data)/batch_size)
        for epoch in range(epochs):
            perm = np.random.permutation(len(data))
            epoch_loss = 0.0
            for b in range(n_batches):
                idx = perm[b*batch_size:(b+1)*batch_size]
                batch = torch.tensor(data[idx], dtype=torch.float32, device=DEVICE)
                loss = rbm.contrastive_divergence(batch, lr=lr)
                epoch_loss += loss
            epoch_loss /= max(1, n_batches)
            if (epoch+1) % max(1, epochs//4) == 0:
                print(f"  Epoch {epoch+1}/{epochs} loss {epoch_loss:.4f}")
        data = rbm.transform(data)
        rbms.append(rbm)
    return rbms

def finetune_dbn(rbms, X_train, y_train, X_val, y_val, finetune_epochs=20, lr=1e-3, batch_size=64):
    layer_sizes = [X_train.shape[1]] + [r.n_hid for r in rbms]
    model = DBNClassifier(layer_sizes, n_classes=2).to(DEVICE)
    # copy weights
    ff_layers = [l for l in model.net if isinstance(l, nn.Linear)]
    for i, rbm in enumerate(rbms):
        W = rbm.W.detach().cpu().numpy()  # shape (n_hid, n_vis)
        b = rbm.h_bias.detach().cpu().numpy()
        # ff_layers[i] maps input->hidden (out_features, in_features)
        ff_layers[i].weight.data = torch.tensor(W, dtype=torch.float32, device=DEVICE)
        ff_layers[i].bias.data = torch.tensor(b, dtype=torch.float32, device=DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n_batches = math.ceil(len(X_train) / batch_size)
    for epoch in range(finetune_epochs):
        perm = np.random.permutation(len(X_train))
        model.train()
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = perm[b*batch_size:(b+1)*batch_size]
            xb = torch.tensor(X_train[idx], dtype=torch.float32, device=DEVICE)
            yb = torch.tensor(y_train[idx], dtype=torch.long, device=DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(1, n_batches)
        # validation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val, dtype=torch.float32, device=DEVICE))
            probs = torch.softmax(val_logits, dim=1)[:,1].cpu().numpy()
            pred = (probs >= 0.5).astype(int)
            auc = roc_auc_score(y_val, probs) if len(np.unique(y_val))>1 else float("nan")
            acc = accuracy_score(y_val, pred)
        if (epoch+1) % max(1, finetune_epochs//5) == 0:
            print(f"Finetune epoch {epoch+1}/{finetune_epochs} loss {epoch_loss:.4f} val_acc {acc:.4f} val_auc {auc:.4f}")
    return model

def main():
    X, y = load_data(CSV_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print("Data shapes:", X_train.shape, X_val.shape)
    rbm_sizes = [128, 64]  # you can change
    rbms = pretrain_rbms(X_train, rbm_sizes, epochs=8, lr=0.01, batch_size=64)
    model = finetune_dbn(rbms, X_train, y_train, X_val, y_val, finetune_epochs=20, lr=1e-3, batch_size=64)
    # Save states
    torch.save([r.state_dict() for r in rbms], RBMS_STATE)
    torch.save(model.state_dict(), CLASS_STATE)
    # Save metadata
    joblib.dump({"rbm_sizes": [r.n_hid for r in rbms], "input_dim": X.shape[1]}, os.path.join(MODEL_DIR, "meta.joblib"))
    print("Saved models to", MODEL_DIR)
    # Evaluate on val
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_val, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        pred = (probs >= 0.5).astype(int)
        print("Val acc:", accuracy_score(y_val, pred))
        try:
            print("Val auc:", roc_auc_score(y_val, probs))
        except:
            pass
        print(classification_report(y_val, pred))

if __name__ == "__main__":
    main()
