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
# 🟢 PHASE 5: FEATURE SCALING
# =================================
st.header("🟢 Phase 5: Feature Scaling")

st.markdown("""
Bevor Machine Learning angewendet wird, müssen alle Features skaliert werden.

👉 Warum?
- unterschiedliche Einheiten (MeV, GeV, Wahrscheinlichkeiten)
- sonst dominiert eine Variable (z. B. Masse)

Wir standardisieren alle Werte auf:
- Mittelwert = 0
- Standardabweichung = 1
""")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Features erfolgreich standardisiert")

# =================================
# 🤖 PHASE 6: KI MODELLVERGLEICH
# =================================
st.header("🤖 Phase 6: Vergleich zweier KI-Modelle")

st.markdown("""
In dieser Phase vergleichen wir zwei verschiedene Ansätze zur Anomalieerkennung:

### 🌲 Isolation Forest
- basiert auf Entscheidungsbäumen
- isoliert ungewöhnliche Punkte schnell

### 🧠 Autoencoder
- neuronales Netzwerk
- lernt typische Muster der Daten
- erkennt Abweichungen über Rekonstruktionsfehler
""")

col1, col2 = st.columns(2)

# =========================================================
# 🌲 ISOLATION FOREST (LINKS)
# =========================================================
with col1:

    st.subheader("🌲 Isolation Forest")

    st.markdown("""
    Dieser Algorithmus prüft:
    👉 Wie leicht lässt sich ein Punkt isolieren?

    Je schneller ein Punkt isoliert wird,
    desto wahrscheinlicher ist er eine Anomalie.
    """)

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    iso_model.fit(X_scaled)
    iso_pred = iso_model.predict(X_scaled)

    # Ergebnis speichern
    iso_df = X.copy()
    iso_df["IsolationForest_Label"] = iso_pred

    iso_anomalies = np.sum(iso_pred == -1)

    st.write("### Ergebnisse")
    st.write(f"🔴 Anomalien erkannt: {iso_anomalies}")
    st.write(f"🟢 Normale Events: {np.sum(iso_pred == 1)}")

    # Visualisierung
    if "Bplus_M" in iso_df.columns:

        st.markdown("### Visualisierung im Feature-Raum")

        fig1, ax1 = plt.subplots(figsize=(6, 4))

        ax1.scatter(
            iso_df["Bplus_M"],
            iso_df["Bplus_PT"],
            c=iso_pred,
            s=5
        )

        ax1.set_title("Isolation Forest Ergebnis")
        ax1.set_xlabel("Bplus_M (Masse)")
        ax1.set_ylabel("Bplus_PT (Impuls)")

        st.pyplot(fig1)

# =========================================================
# 🧠 AUTOENCODER (RECHTS)
# =========================================================
with col2:

    st.subheader("🧠 Autoencoder")

    st.markdown("""
    Dieser Algorithmus funktioniert anders:

    👉 Er lernt, normale Ereignisse zu rekonstruieren.

    Wenn die Rekonstruktion schlecht ist → Anomalie.
    """)

    input_dim = X_scaled.shape[1]

    # Netzwerkarchitektur
    input_layer = Input(shape=(input_dim,))

    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded)

    decoded = Dense(16, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    # Training
    st.markdown("### Training des Modells")

    with st.spinner("Autoencoder lernt Muster der Daten..."):

        autoencoder.fit(
            X_scaled,
            X_scaled,
            epochs=20,
            batch_size=256,
            validation_split=0.2,
            verbose=0
        )

    # Rekonstruktion
    reconstructed = autoencoder.predict(X_scaled, verbose=0)

    # Fehlerberechnung
    reconstruction_error = np.mean(
        np.power(X_scaled - reconstructed, 2),
        axis=1
    )

    # Schwellenwert
    threshold = reconstruction_error.mean() + 2 * reconstruction_error.std()

    ae_pred = reconstruction_error > threshold

    st.markdown("### Ergebnisse")
    st.write(f"🔴 Anomalien erkannt: {np.sum(ae_pred)}")
    st.write(f"🟢 Normale Events: {len(ae_pred) - np.sum(ae_pred)}")

    # Visualisierung
    if "Bplus_M" in X.columns:

        st.markdown("### Rekonstruktionsbasierte Anomalien")

        fig2, ax2 = plt.subplots(figsize=(6, 4))

        ax2.scatter(
            X["Bplus_M"],
            X["Bplus_PT"],
            c=ae_pred,
            s=5
        )

        ax2.set_title("Autoencoder Ergebnis")
        ax2.set_xlabel("Bplus_M (Masse)")
        ax2.set_ylabel("Bplus_PT (Impuls)")

        st.pyplot(fig2)

# =================================
# 📊 PHASE 7: VERGLEICHSANALYSE
# =================================
st.header("📊 Phase 7: Wissenschaftlicher Vergleich")

st.markdown("""
Jetzt vergleichen wir beide Modelle direkt:

👉 Ziel:
- erkennen beide dieselben Anomalien?
- wo unterscheiden sie sich?
- welches Modell reagiert anders?
""")

iso_mask = iso_pred == -1
ae_mask = ae_pred

both = np.sum(iso_mask & ae_mask)
only_iso = np.sum(iso_mask & ~ae_mask)
only_ae = np.sum(~iso_mask & ae_mask)

total = len(X)

st.markdown("### Vergleich der Ergebnisse")

st.write(f"🔁 Beide Modelle erkennen: {both} ({both/total:.2%})")
st.write(f"🌲 Nur Isolation Forest: {only_iso}")
st.write(f"🧠 Nur Autoencoder: {only_ae}")

agreement = (iso_mask == ae_mask).mean()

st.write(f"📏 Gesamt-Übereinstimmung: {agreement:.2%}")
