import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

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
        contamination = st.slider(
            "Wie viele Anomalien?",
            0.01, 0.2, 0.05
        )

        model = IsolationForest(
            contamination=contamination,
            random_state=42
        )

        model.fit(X)
        preds = model.predict(X)

        X = X.copy()
        X["Anomalie"] = preds

        normal = (X["Anomalie"] == 1).sum()
        anomaly = (X["Anomalie"] == -1).sum()

        st.write(f"🟢 Normale Events: {normal}")
        st.write(f"🔴 Anomalien: {anomaly}")

        # -----------------------------
        # 6. VISUALISIERUNG
        # -----------------------------
        st.header("🟣 Phase 6: Visualisierung")

        numeric_cols = X.select_dtypes(include="number").columns

        if len(numeric_cols) >= 2:
            plot_df = X[numeric_cols[:2]]

            st.scatter_chart(plot_df)

        st.subheader("📊 Verteilung")
        st.bar_chart(X["Anomalie"].value_counts())
