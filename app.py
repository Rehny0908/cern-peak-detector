import streamlit as st
import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# =====================================
# STREAMLIT CONFIG
# =====================================
st.set_page_config(
    page_title="CERN KI Analyzer",
    layout="wide"
)

st.title("🧪 KI-gestützte Analyse von CERN-Daten")

# =====================================
# ERKLÄRUNG
# =====================================
st.header("📘 Was macht diese App?")

st.markdown("""
Diese App analysiert Daten aus Teilchenkollisionen.

### Ziele:
- große Datenmengen reduzieren
- normale Ereignisse erkennen
- ungewöhnliche Ereignisse (Anomalien) finden
- KI zur Teilchenanalyse verwenden

Die Daten stammen aus CERN ROOT-Dateien.
""")

# =====================================
# DATEN ERKLÄRUNG
# =====================================
st.header("📊 Physikalische Variablen")

st.markdown("""
- **Bplus_PT** → Transversalimpuls des B+ Teilchens
- **Bplus_M / Bplus_MM** → Masse des Teilchens
- **Bplus_IPCHI2_OWNPV** → Abstand zum Kollisionspunkt
- **Bplus_FDCHI2_OWNPV** → Flugdistanz
- **muplus_PT / muminus_PT** → Impuls der Myonen

Diese Werte helfen der KI dabei, ungewöhnliche Ereignisse zu erkennen.
""")

# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙ Einstellungen")

sample_size = st.sidebar.slider(
    "Stichprobengröße",
    1000,
    50000,
    10000,
    step=1000
)

contamination = st.sidebar.slider(
    "Anteil Anomalien",
    0.01,
    0.20,
    0.05
)

# =====================================
# FILE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "📁 ROOT- oder CSV-Datei hochladen",
    type=["csv", "root"]
)

# =====================================
# DATEI VERARBEITEN
# =====================================
if uploaded_file is not None:

    st.header("🔵 Phase 1: Rohdaten")

    df = None

    # =================================
    # CSV
    # =================================
    if uploaded_file.name.endswith(".csv"):

        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV-Datei erfolgreich geladen")

        except Exception as e:
            st.error(f"Fehler beim Laden der CSV-Datei: {e}")
            st.stop()

    # =================================
    # ROOT
    # =================================
    elif uploaded_file.name.endswith(".root"):

        st.info("ROOT-Datei erkannt")

        try:
            # ROOT öffnen
            file = uproot.open(uploaded_file)

            st.write("📂 ROOT Keys:")
            st.write(file.keys())

            # Tree auswählen
            tree = file["Btree/DecayTree"]

            st.success("DecayTree gefunden")

            # Features
            root_features = [
                "Bplus_PT",
                "Bplus_MM",
                "Bplus_IPCHI2_OWNPV",
                "Bplus_FDCHI2_OWNPV",
                "Bplus_DIRA_OWNPV",
                "muplus_PT",
                "muminus_PT"
            ]

            # existierende Features
            available = [
                f for f in root_features
                if f in tree.keys()
            ]

            st.write("✔ Verfügbare Features:")
            st.write(available)

            # ROOT -> pandas
            df = tree.arrays(available, library="pd")

            # bessere Namensgleichheit
            if "Bplus_MM" in df.columns:
                df["Bplus_M"] = df["Bplus_MM"]

            st.success("ROOT-Datei erfolgreich geladen")

        except Exception as e:
            st.error(f"Fehler beim Laden der ROOT-Datei: {e}")
            st.stop()

    # =================================
    # DATEN CHECK
    # =================================
    if df is not None:

        st.write(f"📦 Ursprüngliche Events: {len(df):,}")

        # Sampling
        if len(df) > sample_size:
            df = df.sample(
                n=sample_size,
                random_state=42
            )

        st.write(f"📉 Verwendete Events: {len(df):,}")

        st.dataframe(df.head())

        # =================================
        # MASSENPEAK
        # =================================
        if "Bplus_M" in df.columns:

            st.header("🟣 Physikalischer Mass-Peak")

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(
                df["Bplus_M"],
                bins=100
            )

            ax.set_xlabel("Masse (MeV)")
            ax.set_ylabel("Events")
            ax.set_title("B+ Mass Peak")

            st.pyplot(fig)

            st.markdown("""
Der Peak bei ungefähr 5300 MeV entspricht vermutlich echten B+ Zerfällen.

Bereiche außerhalb des Peaks sind meist Hintergrundereignisse.
""")

        # =================================
        # CLEANING
        # =================================
        st.header("🟡 Phase 2: Datenbereinigung")

        before = len(df)

        df = df.dropna()

        after = len(df)

        st.write(f"Vorher: {before:,}")
        st.write(f"Nachher: {after:,}")
        st.write(f"❌ Entfernt: {before - after:,}")

        # =================================
        # FEATURE AUSWAHL
        # =================================
        st.header("🟠 Phase 3: Feature Auswahl")

        features = [
            "Bplus_PT",
            "Bplus_M",
            "Bplus_IPCHI2_OWNPV",
            "Bplus_FDCHI2_OWNPV",
            "Bplus_DIRA_OWNPV",
            "muplus_PT",
            "muminus_PT"
        ]

        available_features = [
            f for f in features
            if f in df.columns
        ]

        st.write("✔ Genutzte Features:")
        st.write(available_features)

        if len(available_features) < 2:
            st.error("Nicht genug Features verfügbar")
            st.stop()

        X = df[available_features]

        # =================================
        # PHYSIK FILTER
        # =================================
        st.header("🟠 Phase 4: Physikalische Filter")

        if "Bplus_PT" in X.columns:

            pt_cut = st.slider(
                "Minimaler Bplus_PT",
                float(X["Bplus_PT"].min()),
                float(X["Bplus_PT"].max()),
                float(X["Bplus_PT"].median())
            )

            before_filter = len(X)

            X = X[X["Bplus_PT"] > pt_cut]

            after_filter = len(X)

            st.write(
                f"📉 Nach PT-Filter: {after_filter:,} "
                f"({before_filter - after_filter:,} entfernt)"
            )

# =================================
# 🟢 FEATURE SCALING
# =================================
st.header("🟢 Feature Scaling")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Scaling abgeschlossen")


# =================================
# 🤖 MODELLVERGLEICH
# =================================
st.header("🤖 KI Modellvergleich")

col1, col2 = st.columns(2)

# =========================================================
# 🌲 ISOLATION FOREST
# =========================================================
with col1:

    st.subheader("🌲 Isolation Forest")

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    iso_pred = iso_model.fit_predict(X_scaled)

    iso_anom = (iso_pred == -1).sum()

    st.metric("Anomalien", iso_anom)

    iso_df = X.copy()
    iso_df["label"] = iso_pred

    if "Bplus_M" in X.columns:

        fig1, ax1 = plt.subplots()

        ax1.scatter(
            X["Bplus_M"],
            X["Bplus_PT"],
            c=iso_pred,
            s=5
        )

        ax1.set_title("Isolation Forest")

        st.pyplot(fig1)


# =========================================================
# 🧠 AUTOENCODER
# =========================================================
with col2:

    st.subheader("🧠 Autoencoder")

    input_dim = X_scaled.shape[1]

    inp = Input(shape=(input_dim,))
    x = Dense(16, activation="relu")(inp)
    x = Dense(8, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)

    autoencoder = Model(inp, out)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        verbose=0
    )

    recon = autoencoder.predict(X_scaled, verbose=0)

    mse = np.mean(np.square(X_scaled - recon), axis=1)

    threshold = np.percentile(mse, 95)
    ae_pred = mse > threshold

    ae_anom = ae_pred.sum()

    st.metric("Anomalien", ae_anom)

    if "Bplus_M" in X.columns:

        fig2, ax2 = plt.subplots()

        ax2.scatter(
            X["Bplus_M"],
            X["Bplus_PT"],
            c=ae_pred,
            s=5
        )

        ax2.set_title("Autoencoder")

        st.pyplot(fig2)


# =================================
# 📊 VERGLEICH
# =================================
st.header("📊 Vergleich")

iso_mask = iso_pred == -1
ae_mask = ae_pred

both = (iso_mask & ae_mask).sum()
only_iso = (iso_mask & ~ae_mask).sum()
only_ae = (~iso_mask & ae_mask).sum()

total = len(X)

st.write(f"Overlap: {both} ({both/total:.2%})")
st.write(f"Nur IF: {only_iso}")
st.write(f"Nur AE: {only_ae}")

agreement = (iso_mask == ae_mask).mean()

st.metric("Übereinstimmung", f"{agreement:.2%}")
