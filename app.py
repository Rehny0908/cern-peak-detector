import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers

st.title("CERN Anomalie Detektor (Eigenes Modell)")

uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Features auswählen
    features = [
        "Bplus_PT",
        "Bplus_M",
        "Bplus_IPCHI2_OWNPV",
        "muplus_PT",
        "muplus_ProbNNmu",
        "muminus_PT",
        "muminus_ProbNNmu"
    ]

    df = df[features].dropna()

    st.write(f"Anzahl Events: {len(df)}")

    # Normalisieren
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # Autoencoder bauen
    model = tf.keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
        layers.Dense(8, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(X.shape[1])
    ])

    model.compile(optimizer='adam', loss='mse')

    # Training
    with st.spinner("Trainiere Modell..."):
        model.fit(X, X, epochs=10, batch_size=32, verbose=0)

    # Rekonstruktion
    recon = model.predict(X)

    # Fehler berechnen
    loss = tf.reduce_mean(tf.square(X - recon), axis=1)

    # Schwelle definieren
    threshold = loss.numpy().mean() + loss.numpy().std()

    df["Fehler"] = loss.numpy()
    df["Anomalie"] = df["Fehler"] > threshold

    # Anzeige
    st.write("Ergebnisse:")
    st.write(df.head())

    st.write("Anzahl Anomalien:")
    st.write(df["Anomalie"].value_counts())
