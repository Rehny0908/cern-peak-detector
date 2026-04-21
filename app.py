import streamlit as st
import pandas as pd

st.set_page_config(page_title="CERN Data Filter", layout="wide")

st.title("🧪 CERN Daten-Reduktions-Tool")

uploaded_file = st.file_uploader("CSV hochladen", type=["csv"])

if uploaded_file:

    # -----------------------------
    # 1. DATEN LADEN (nur Teil!)
    # -----------------------------
    st.header("🔵 Phase 1: Daten laden")

    df = pd.read_csv(uploaded_file)

    st.write(f"📦 Originalgröße: {len(df):,} Events")

    # SAMPLE (entscheidend!)
    sample_size = st.slider("Stichprobengröße", 1000, 50000, 10000)

    df = df.sample(n=sample_size, random_state=42)

    st.write(f"📉 Verwendete Stichprobe: {len(df):,}")

    # -----------------------------
    # 2. NA FILTER
    # -----------------------------
    st.header("🟡 Phase 2: Fehlmessungen entfernen")

    before = len(df)
    df = df.dropna()
    after = len(df)

    st.write(f"Vorher: {before:,}")
    st.write(f"Nachher: {after:,}")
    st.write(f"❌ Entfernt: {before - after:,}")

    # -----------------------------
    # 3. PHYSIK FILTER
    # -----------------------------
    st.header("🟠 Phase 3: Physikalische Filter")

    # sichere Spalten
    cols = df.columns

    if "Bplus_PT" in cols:
        pt_cut = st.slider("Bplus_PT Mindestwert", 0.0, float(df["Bplus_PT"].max()), 500.0)

        before = len(df)
        df = df[df["Bplus_PT"] > pt_cut]
        after = len(df)

        st.write(f"📉 Nach PT-Filter: {after:,} ({before-after:,} entfernt)")

    if "Bplus_M" in cols:
        mass_min = st.slider("Masse min", float(df["Bplus_M"].min()), float(df["Bplus_M"].max()), float(df["Bplus_M"].min()))
        mass_max = st.slider("Masse max", float(df["Bplus_M"].min()), float(df["Bplus_M"].max()), float(df["Bplus_M"].max()))

        before = len(df)
        df = df[(df["Bplus_M"] > mass_min) & (df["Bplus_M"] < mass_max)]
        after = len(df)

        st.write(f"📉 Nach Massenfilter: {after:,} ({before-after:,} entfernt)")

    # -----------------------------
    # 4. ERGEBNIS
    # -----------------------------
    st.header("🟢 Phase 4: Ergebnis")

    st.write(f"📦 Übrig: {len(df):,} Events")

    st.write(df.head())

    # einfache Visualisierung
    st.subheader("📈 Übersicht")

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) >= 2:
        st.scatter_chart(df[numeric_cols[:2]])
