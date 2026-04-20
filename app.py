import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.title("CERN Anomaly Detector")

uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])

if uploaded_file:

    # CSV laden
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Datenübersicht")
    st.write(df.head())

    st.write("📌 Spalten im Datensatz:")
    st.write(df.columns.tolist())

    # gewünschte Features (aus deinem CERN-Setup)
    features = [
        "Bplus_PT",
        "Bplus_M",
        "Bplus_IPCHI2_OWNPV",
        "muplus_PT",
        "muminus_PT"
    ]

    # nur existierende Spalten nutzen (verhindert KeyError!)
    available_features = [f for f in features if f in df.columns]

    if len(available_features) == 0:
        st.error("❌ Keine passenden Features in der CSV gefunden!")
        st.stop()

    st.success(f"✔ Genutzte Features: {available_features}")

    # Daten vorbereiten
    X = df[available_features].dropna()

    st.write(f"📦 Verwendete Zeilen: {len(X)}")

    # Modell
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)

    # Vorhersage
    preds = model.predict(X)

    # -1 = Anomalie, 1 = normal
    X = X.copy()
    X["Anomalie"] = preds

    # Anzeige
    st.subheader("🔎 Ergebnisse")
    st.write(X.head())

    st.subheader("📊 Verteilung")
    st.write(X["Anomalie"].value_counts())

    # einfache Visualisierung
    st.subheader("📈 Beispiel Plot")
    st.scatter_chart(X[[available_features[0], available_features[1]]])
