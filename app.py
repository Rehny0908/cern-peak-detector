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

st.title("🧪 KI-gestützte Analyse von CERN-Teilchendaten")


# =====================================
# PROJEKTBESCHREIBUNG
# =====================================
st.header("📘 Projektbeschreibung")

st.markdown("""
Diese Anwendung analysiert reale CERN-Kollisionsdaten aus ROOT-Dateien.

Ziel ist die Identifikation seltener oder ungewöhnlicher Ereignisse
mithilfe unüberwachter Machine-Learning-Verfahren.

Dabei werden zwei unterschiedliche KI-Ansätze systematisch verglichen.
""")


# =====================================
# PHYSIKALISCHE FEATURES
# =====================================
st.header("📊 Physikalische Variablen")

st.markdown("""
Die Analyse basiert auf rekonstruierten Eigenschaften von B⁺-Mesonen:

- **Bplus_PT** → Transversalimpuls des Teilchens  
- **Bplus_M / Bplus_MM** → Invariante Masse des Kandidaten  
- **Bplus_IPCHI2_OWNPV** → Impact Parameter χ² zum Primärvertex  
- **Bplus_FDCHI2_OWNPV** → Flugdistanzsignifikanz  
- **muplus_PT / muminus_PT** → Impuls der Zerfallsprodukte  

Diese Größen beschreiben die kinematische Struktur der Ereignisse.
""")


# =====================================
# SIDEBAR
# =====================================
st.sidebar.header("⚙ Konfiguration")

sample_size = st.sidebar.slider(
    "Stichprobengröße",
    1000,
    50000,
    10000,
    step=1000
)

contamination = st.sidebar.slider(
    "Erwarteter Anomalieanteil",
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
# DATENVERARBEITUNG
# =====================================
if uploaded_file is not None:

    df = None

    # ---------------- CSV ----------------
    if uploaded_file.name.endswith(".csv"):

        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV erfolgreich geladen")
        except Exception as e:
            st.error(f"Fehler beim CSV-Laden: {e}")
            st.stop()


    # ---------------- ROOT ----------------
    elif uploaded_file.name.endswith(".root"):

        st.info("ROOT-Datei erkannt")

        try:
            file = uproot.open(uploaded_file)

            tree = file["Btree/DecayTree"]

            st.write("Verfügbare ROOT-Keys:", file.keys())

            features = [
                "Bplus_PT",
                "Bplus_MM",
                "Bplus_IPCHI2_OWNPV",
                "Bplus_FDCHI2_OWNPV",
                "Bplus_DIRA_OWNPV",
                "muplus_PT",
                "muminus_PT"
            ]

            available = [f for f in features if f in tree.keys()]

            df = tree.arrays(available, library="pd")

            if "Bplus_MM" in df.columns:
                df["Bplus_M"] = df["Bplus_MM"]

            st.success("ROOT erfolgreich geladen")

        except Exception as e:
            st.error(f"ROOT Fehler: {e}")
            st.stop()


    # =====================================
    # DATENBASIS
    # =====================================
    if df is not None:

        st.write(f"Ursprüngliche Events: {len(df):,}")

        df = df.sample(min(len(df), sample_size), random_state=42)
        df = df.dropna()

        st.write(f"Verwendete Events: {len(df):,}")

        st.dataframe(df.head())


        # =====================================
        # MASSENVERTEILUNG
        # =====================================
        if "Bplus_M" in df.columns:

            st.header("🟣 Invariante Massenverteilung")

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.hist(df["Bplus_M"], bins=100)

            ax.set_xlabel("Masse [MeV]")
            ax.set_ylabel("Anzahl Events")
            ax.set_title("Rekonstruierte B⁺-Masse")

            st.pyplot(fig)

            st.markdown("""
Der Peak bei ca. 5300 MeV entspricht rekonstruierten B⁺-Zerfällen.

Die restliche Verteilung beschreibt Hintergrundereignisse.
""")


        # =====================================
        # FEATURE SELECTION
        # =====================================
        st.header("🟡 Feature-Auswahl")

        features = [
            "Bplus_PT",
            "Bplus_M",
            "Bplus_IPCHI2_OWNPV",
            "Bplus_FDCHI2_OWNPV",
            "Bplus_DIRA_OWNPV",
            "muplus_PT",
            "muminus_PT"
        ]

        available_features = [f for f in features if f in df.columns]

        X = df[available_features]

        st.write("Verwendete Features:", available_features)


        # =====================================
        # PHYSIK FILTER
        # =====================================
        st.header("🟠 Physikalischer Filter")

        if "Bplus_PT" in X.columns:

            pt_cut = st.slider(
                "PT Cut",
                float(X["Bplus_PT"].min()),
                float(X["Bplus_PT"].max()),
                float(X["Bplus_PT"].median())
            )

            X = X[X["Bplus_PT"] > pt_cut]


        # =====================================
        # SCALING
        # =====================================
        st.header("🟢 Feature Scaling")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.success("Standardisierung abgeschlossen")


        # =====================================
        # MODELLVERGLEICH
        # =====================================
        st.header("🤖 Vergleich der KI-Modelle")

        st.markdown("""
Zwei unüberwachte Verfahren werden verglichen:

- Isolation Forest: strukturbasierte Ausreißererkennung
- Autoencoder: rekonstruktionsbasierte Fehlerdetektion
""")

        col1, col2 = st.columns(2)


        # =====================================
        # ISOLATION FOREST
        # =====================================
        with col1:

            st.subheader("🌲 Isolation Forest")

            iso_model = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42
            )

            iso_pred = iso_model.fit_predict(X_scaled)

            iso_anom = (iso_pred == -1).sum()

            st.write(f"Anomalien: {iso_anom}")

            iso_df = X.copy()
            iso_df["label"] = iso_pred

            if "Bplus_M" in X.columns:

                fig, ax = plt.subplots()

                ax.scatter(
                    X["Bplus_M"],
                    X["Bplus_PT"],
                    c=iso_pred,
                    s=5
                )

                ax.set_title("Isolation Forest")

                st.pyplot(fig)


        # =====================================
        # AUTOENCODER
        # =====================================
        with col2:

            st.subheader("🧠 Autoencoder")

            input_dim = X_scaled.shape[1]

            inp = Input(shape=(input_dim,))
            x = Dense(16, activation="relu")(inp)
            x = Dense(8, activation="relu")(x)
            out = Dense(input_dim, activation="linear")(x)

            model = Model(inp, out)
            model.compile(optimizer="adam", loss="mse")

            model.fit(
                X_scaled,
                X_scaled,
                epochs=20,
                batch_size=256,
                validation_split=0.2,
                verbose=0
            )

            recon = model.predict(X_scaled)

            mse = np.mean(np.square(X_scaled - recon), axis=1)

            threshold = np.mean(mse) + 2 * np.std(mse)

            ae_pred = mse > threshold

            st.write(f"Anomalien: {ae_pred.sum()}")

            if "Bplus_M" in X.columns:

                fig, ax = plt.subplots()

                ax.scatter(
                    X["Bplus_M"],
                    X["Bplus_PT"],
                    c=ae_pred,
                    s=5
                )

                ax.set_title("Autoencoder")

                st.pyplot(fig)


        # =====================================
        # VERGLEICH
        # =====================================
        st.header("📊 Ergebnisvergleich")

        iso_mask = iso_pred == -1
        ae_mask = ae_pred

        both = (iso_mask & ae_mask).sum()
        only_iso = (iso_mask & ~ae_mask).sum()
        only_ae = (~iso_mask & ae_mask).sum()

        total = len(X)

        st.write(f"Beide Modelle: {both} ({both/total:.2%})")
        st.write(f"Nur Isolation Forest: {only_iso}")
        st.write(f"Nur Autoencoder: {only_ae}")

        agreement = (iso_mask == ae_mask).mean()

        st.write(f"Übereinstimmung: {agreement:.2%}")
