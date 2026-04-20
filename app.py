import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="CERN AI Analyzer", layout="wide")

st.title("🧪 CERN Event Anomaly Analyzer")

# -----------------------------
# 1. UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📁 CSV hochladen", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.header("🔵 Phase 1: Rohdaten")
    st.write(df.head())
    st.write(f"📦 Gesamt Events: {len(df):,}")

    # -----------------------------
    # 2. CLEANING
    # -----------------------------
    st.header("🟡 Phase 2: Datenbereinigung")

    cleaning_mode = st.radio(
        "Umgang mit fehlenden Werten",
        ["NaNs entfernen", "Mit Median füllen"]
    )

    if cleaning_mode == "NaNs entfernen":
        df_clean = df.dropna()
    else:
        df_clean = df.fillna(df.median(numeric_only=True))

    st.write(f"📦 Nach Cleaning: {len(df_clean):,}")

    # -----------------------------
    # 3. FEATURE SELECTION
    # -----------------------------
    st.header("🟠 Phase 3: Physikalische Features")

    features = [
        "Bplus_PT",
        "Bplus_M",
        "Bplus_IPCHI2_OWNPV",
        "muplus_PT",
        "muminus_PT"
    ]

    available_features = [f for f in features if f in df_clean.columns]

    st.write("✔ Verwendete Features:")
    st.write(available_features)

    if len(available_features) < 2:
        st.error("Zu wenige Features vorhanden!")
        st.stop()

    X = df_clean[available_features]

    # -----------------------------
    # 4. KI MODELL
    # -----------------------------
    st.header("🔴 Phase 4: KI Anomalie-Erkennung")

    contamination = st.slider(
        "Anomalie-Sensitivität",
        0.01, 0.2, 0.05
    )

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    model.fit(X)

    preds = model.predict(X)

    result = X.copy()
    result["Anomaly"] = preds

    st.write(result.head())

    normal = (result["Anomaly"] == 1).sum()
    anomaly = (result["Anomaly"] == -1).sum()

    st.metric("Normale Events", normal)
    st.metric("Anomalien", anomaly)

    # -----------------------------
    # 5. VISUALISIERUNG
    # -----------------------------
    st.header("🟣 Phase 5: Visualisierung")

    plot_df = result.copy()

    if len(available_features) >= 2:
        st.subheader("Feature Space")

        chart_data = plot_df[[available_features[0], available_features[1], "Anomaly"]]

        st.scatter_chart(
            chart_data,
            x=available_features[0],
            y=available_features[1]
        )

    st.subheader("📊 Verteilung")

    st.bar_chart(result["Anomaly"].value_counts())
