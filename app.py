import streamlit as st
import pandas as pd
import numpy as np
import uproot
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
        # SCALING
        # =================================
        st.header("🟢 Phase 5: Feature Scaling")

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        st.success("Features erfolgreich skaliert")

        # =================================
        # ISOLATION FOREST
        # =================================
        st.header("🔴 Phase 6: Isolation Forest")

        if len(X_scaled) < 100:

            st.warning("Zu wenige Daten für KI")

        else:

            model = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42
            )

            model.fit(X_scaled)

            predictions = model.predict(X_scaled)

            X_result = X.copy()

            X_result["Anomalie"] = predictions

            normal = (predictions == 1).sum()
            anomaly = (predictions == -1).sum()

            st.write(f"🟢 Normale Events: {normal}")
            st.write(f"🔴 Anomalien: {anomaly}")

            # =================================
            # SCATTER PLOT
            # =================================
            st.header("🟣 Phase 7: KI Visualisierung")

            if "Bplus_M" in X_result.columns and "Bplus_PT" in X_result.columns:

                fig2, ax2 = plt.subplots(figsize=(10, 6))

                normal_data = X_result[X_result["Anomalie"] == 1]
                anomaly_data = X_result[X_result["Anomalie"] == -1]

                ax2.scatter(
                    normal_data["Bplus_M"],
                    normal_data["Bplus_PT"],
                    s=5,
                    label="Normal"
                )

                ax2.scatter(
                    anomaly_data["Bplus_M"],
                    anomaly_data["Bplus_PT"],
                    s=10,
                    label="Anomalie"
                )

                ax2.set_xlabel("Bplus_M")
                ax2.set_ylabel("Bplus_PT")
                ax2.set_title("Isolation Forest Ergebnisse")

                ax2.legend()

                st.pyplot(fig2)

            # =================================
            # BAR CHART
            # =================================
            st.subheader("📊 Verteilung")

            st.bar_chart(
                X_result["Anomalie"].value_counts()
            )

            # =================================
            # ANOMALIEN TABELLE
            # =================================
            st.header("📋 Gefundene Anomalien")

            anomaly_df = X_result[
                X_result["Anomalie"] == -1
            ]

            st.dataframe(anomaly_df.head(20))
