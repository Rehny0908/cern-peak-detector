import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

import joblib, json, torch
# --- Autoencoder: Klasse (muss mit train_autoencoder kompatibel sein) ---
import torch.nn as nn
from torch import Tensor

# Duplikat der AE-Klasse (identisch zum Training)
class AE(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        h = max(8, n_features*2//3)
        latent = max(2, n_features // 3)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, h),
            nn.ReLU(),
            nn.Linear(h, latent),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, h),
            nn.ReLU(),
            nn.Linear(h, n_features)
        )
    def forward(self, x: Tensor):
        return self.decoder(self.encoder(x))

# --- Pfade zu deinen gespeicherten Artefakten ---
AE_SCALER = "scaler.joblib"
AE_MODEL = "ae_model.pt"
AE_META = "ae_meta.json"

# lade falls vorhanden
ae_artifacts_loaded = False
try:
    ae_scaler = joblib.load(AE_SCALER)
    ae_meta = json.load(open(AE_META))
    # AE-Instanz wird später mit korrekter n_features erzeugt
    ae_artifacts_loaded = True
except Exception:
    ae_artifacts_loaded = False


st.set_page_config(page_title="CERN KI Analyzer", layout="wide")

st.title("🧪 KI-gestützte Analyse von CERN-Daten")

# -----------------------------
# ERKLÄRUNG
# -----------------------------
st.header("📘 Was macht diese App?")

st.markdown("""
Diese App analysiert Daten aus Teilchenkollisionen.

👉 Ziel:
- große Datenmengen reduzieren  
- normale Ereignisse erkennen  
- ungewöhnliche Ereignisse (Anomalien) finden  

Die KI hilft dabei, interessante physikalische Ereignisse zu entdecken.
""")

# -----------------------------
# DATEN ERKLÄRUNG
# -----------------------------
st.header("📊 Was bedeuten die Variablen?")

st.markdown("""
- **Bplus_PT** → Bewegung / Energie des Teilchens  
- **Bplus_M** → Masse des Teilchens  
- **Bplus_IPCHI2_OWNPV** → Abstand vom Kollisionspunkt  
- **muplus_PT / muminus_PT** → Energie der Zerfallsprodukte  

👉 Diese Werte helfen zu entscheiden, ob ein Ereignis „normal“ ist.
""")

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📁 CSV-Datei hochladen", type=["csv"])

if uploaded_file:

    # -----------------------------
    # 1. DATEN LADEN
    # -----------------------------
    st.header("🔵 Phase 1: Rohdaten")

    df = pd.read_csv(uploaded_file)

    st.write(f"📦 Ursprüngliche Events: {len(df):,}")

    # SAMPLE (wichtig für Performance)
    sample_size = st.slider("Stichprobengröße", 1000, 50000, 10000)
    df = df.sample(n=sample_size, random_state=42)

    st.write(f"📉 Verwendete Daten: {len(df):,}")

    # -----------------------------
    # 2. CLEANING
    # -----------------------------
    st.header("🟡 Phase 2: Datenbereinigung")

    before = len(df)
    df = df.dropna()
    after = len(df)

    st.write(f"Vorher: {before:,}")
    st.write(f"Nachher: {after:,}")
    st.write(f"❌ Entfernt: {before - after:,}")

    # -----------------------------
    # 3. FEATURE AUSWAHL
    # -----------------------------
    st.header("🟠 Phase 3: Relevante Daten")

    features = [
        "Bplus_PT",
        "Bplus_M",
        "Bplus_IPCHI2_OWNPV",
        "muplus_PT",
        "muminus_PT"
    ]

    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        st.error("❌ Nicht genug passende Daten!")
        st.stop()

    st.write("✔ Genutzte Features:", available_features)

    X = df[available_features]

    # -----------------------------
    # 4. PHYSIK FILTER
    # -----------------------------
    st.header("🟠 Phase 4: Physikalische Filter")

    if "Bplus_PT" in X.columns:
        pt_cut = st.slider(
            "Mindestenergie (Bplus_PT)",
            float(X["Bplus_PT"].min()),
            float(X["Bplus_PT"].max()),
            float(X["Bplus_PT"].median())
        )

        before = len(X)
        X = X[X["Bplus_PT"] > pt_cut]
        after = len(X)

        st.write(f"📉 Nach Filter: {after:,} ({before - after:,} entfernt)")

    # -----------------------------
    # 5. KI
    # -----------------------------
    st.header("🔴 Phase 5: KI erkennt Anomalien")

    st.markdown("""
Die KI lernt typische Ereignisse und erkennt Abweichungen.

👉 Rot = ungewöhnlich  
👉 Grün = normal  
""")

if len(X) < 100:
    st.warning("Zu wenige Daten für KI")
else:
    # IsolationForest (wie bisher)
    contamination = st.slider("Wie viele Anomalien (IsolationForest)?", 0.01, 0.2, 0.05)
    if_model = IsolationForest(contamination=contamination, random_state=42)
    if_model.fit(X)
    if_preds = if_model.predict(X)
    X = X.copy()
    X["IF_Anomalie"] = if_preds
    normal = (X["IF_Anomalie"] == 1).sum()
    anomaly = (X["IF_Anomalie"] == -1).sum()
    st.write(f"🟢 IF Normale Events: {normal}")
    st.write(f"🔴 IF Anomalien: {anomaly}")

    # Autoencoder (falls Artefakte geladen)
    if ae_artifacts_loaded and all(f in X.columns for f in ae_meta["features"]):
        X_ae = X[ae_meta["features"]].fillna(X[ae_meta["features"]].median()).values.astype("float32")
        X_scaled = ae_scaler.transform(X_ae)
        model = AE(n_features=X_scaled.shape[1])
        model.load_state_dict(torch.load(AE_MODEL, map_location="cpu"))
        model.eval()
        with torch.no_grad():
            recon = model(torch.from_numpy(X_scaled)).numpy()
        ae_error = ((recon - X_scaled)**2).mean(axis=1)
        X["ae_error"] = ae_error
        default_thresh = ae_meta.get("threshold", float(np.percentile(ae_error,95)))
        thresh = st.slider("AE Threshold", float(ae_error.min()), float(ae_error.max()), default_thresh)
        X["AE_Anomalie"] = X["ae_error"] > thresh
        st.write(f"🔴 AE Anomalien: {int(X['AE_Anomalie'].sum())}")
    else:
        st.info("Autoencoder‑Artefakte nicht gefunden oder fehlende Features; nur IsolationForest ausgeführt.")

    # Vergleich & Visualisierung
    st.header("🟣 Phase 6: Vergleich & Visualisierung")
    if "AE_Anomalie" in X.columns:
        both = pd.DataFrame({
            "IF": X["IF_Anomalie"],
            "AE": X["AE_Anomalie"],
            "ae_error": X.get("ae_error", pd.Series([0]*len(X)))
        }, index=X.index)
        st.write("Kontingenztabelle IF vs AE:")
        st.write(pd.crosstab(both["IF"], both["AE"]))
        st.subheader("Top AE Anomalien")
        st.dataframe(X.loc[X["AE_Anomalie"].sort_values(ascending=False).index, ae_meta["features"] + ["ae_error"]].head(20))
    # existing simple plots
    numeric_cols = X.select_dtypes(include="number").columns
    if len(numeric_cols) >= 2:
        plot_df = X[numeric_cols[:2]]
        st.scatter_chart(plot_df)
    st.subheader("📊 IF Verteilung")
    st.bar_chart(X["IF_Anomalie"].value_counts())
