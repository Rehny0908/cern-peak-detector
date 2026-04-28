# train_autoencoder.py
import numpy as np, pandas as pd, joblib, json, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CSV_PATH = "cern_data.csv"   # passe an
FEATURES = ["Bplus_PT","Bplus_M","Bplus_IPCHI2_OWNPV","muplus_PT","muminus_PT"]
BATCH_SIZE = 128; EPOCHS = 30; LR = 1e-3; THRESH_PERCENTILE = 95
OUT_SCALER = "scaler.joblib"; OUT_MODEL = "ae_model.pt"; OUT_META = "ae_meta.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(CSV_PATH)
X = df[FEATURES].copy().fillna(df[FEATURES].median())
X_np = X.values.astype(np.float32)
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X_np)
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val)), batch_size=BATCH_SIZE, shuffle=False)

n_features = X_train.shape[1]; latent = max(2, n_features // 3)
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        h = max(8, n_features*2//3)
        self.encoder = nn.Sequential(nn.Linear(n_features, h), nn.ReLU(), nn.Linear(h, latent), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent, h), nn.ReLU(), nn.Linear(h, n_features))
    def forward(self, x): return self.decoder(self.encoder(x))

model = AE().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=LR); loss_fn = nn.MSELoss()
for ep in range(EPOCHS):
    model.train(); tr=0.0
    for (batch,) in train_loader:
        b = batch.to(DEVICE); recon = model(b); loss = loss_fn(recon,b)
        opt.zero_grad(); loss.backward(); opt.step(); tr += loss.item()*b.size(0)
    model.eval(); val=0.0
    with torch.no_grad():
        for (batch,) in val_loader:
            bb = batch.to(DEVICE); val += loss_fn(model(bb), bb).item()*bb.size(0)
    print(f"Epoch {ep+1}/{EPOCHS} train={tr/len(train_loader.dataset):.6f} val={val/len(val_loader.dataset):.6f}")

with torch.no_grad():
    X_tensor = torch.from_numpy(X_scaled).to(DEVICE)
    recon = model(X_tensor).cpu().numpy()
recon_error = np.mean((recon - X_scaled)**2, axis=1)

with torch.no_grad():
    v_tensor = torch.from_numpy(X_val).to(DEVICE)
    v_recon = model(v_tensor).cpu().numpy()
val_err = np.mean((v_recon - X_val)**2, axis=1)
threshold = float(np.percentile(val_err, THRESH_PERCENTILE))
joblib.dump(scaler, OUT_SCALER); torch.save(model.state_dict(), OUT_MODEL)
json.dump({"features": FEATURES, "threshold": threshold}, open(OUT_META,"w"))
print("Saved:", OUT_SCALER, OUT_MODEL, OUT_META)
