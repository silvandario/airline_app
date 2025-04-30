import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Lade Modelle und Scaler
xgb_model = joblib.load("models/xgb_best_model.pkl")
rf_model = joblib.load("models/rf_best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Feature-Reihenfolge (ohne Zielvariable)
feature_order = [
    'Age', 'Flight Distance', 'Departure and Arrival Time Convenience',
    'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
    'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service',
    'Cleanliness', 'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
    'In-flight Entertainment', 'Baggage Handling', 'Gender_Male',
    'Customer Type_Returning', 'Type of Travel_Personal', 'Class_Economy',
    'Class_Economy Plus', 'Delay'
]

st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide")
st.title("✈️ Airline Satisfaction Prediction & Insights App")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Feature Insights", "📤 CSV Upload", "✍️ Manueller Input"])

# Tab 1: Visualisierung
with tab1:
    st.header("🎯 Feature Importances")
    
    # XGBoost
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_order).sort_values(ascending=True)
    st.subheader("XGBoost")
    fig, ax = plt.subplots(figsize=(10, 6))
    xgb_importance.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title("Top Features - XGBoost")
    st.pyplot(fig)

    # Random Forest
    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_order).sort_values(ascending=True)
    st.subheader("Random Forest")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    rf_importance.plot(kind='barh', ax=ax2, color='darkgreen')
    ax2.set_title("Top Features - Random Forest")
    st.pyplot(fig2)

    st.markdown("---")
    st.header("🔍 Segmentbasierte Analyse")
    filter_class = st.selectbox("Filter nach Klasse", ["Alle", "Economy", "Economy Plus", "Business"])
    filter_type = st.selectbox("Filter nach Reisetyp", ["Alle", "Geschäftlich", "Privat"])
    
    try:
        sample_df = pd.read_csv("sample_data/sample_clean.csv")
    except FileNotFoundError:
        st.warning("⚠️ Beispiel-Datensatz nicht gefunden. Bitte lade eine CSV-Datei hoch.")
        sample_df = pd.DataFrame(columns=feature_order)

    if filter_class != "Alle":
        if filter_class == "Economy":
            sample_df = sample_df[sample_df['Class_Economy'] == 1]
        elif filter_class == "Economy Plus":
            sample_df = sample_df[sample_df['Class_Economy Plus'] == 1]
        else:
            sample_df = sample_df[(sample_df['Class_Economy'] == 0) & (sample_df['Class_Economy Plus'] == 0)]

    if filter_type != "Alle":
        personal = 1 if filter_type == "Privat" else 0
        sample_df = sample_df[sample_df['Type of Travel_Personal'] == personal]

    st.write(f"Gefilterte Datenpunkte: {len(sample_df)}")
    st.dataframe(sample_df.head())

# Tab 2: CSV Upload
with tab2:
    uploaded_file = st.file_uploader("Lade eine CSV-Datei hoch", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if not set(feature_order).issubset(df.columns):
            st.error("Die CSV-Datei enthält nicht alle benötigten Spalten.")
        else:
            df_scaled = scaler.transform(df[feature_order])
            df["XGBoost Prediction"] = xgb_model.predict(df_scaled)
            df["Random Forest Prediction"] = rf_model.predict(df_scaled)
            st.success("Vorhersage erfolgreich durchgeführt!")
            st.dataframe(df.head())

# Tab 3: Manueller Input
with tab3:
    st.subheader("Manuelle Eingabe eines Beispiels")

    def user_input_features():
        values = {}
        values['Age'] = st.slider("Alter", 18, 80, 35)
        values['Flight Distance'] = st.slider("Flugdistanz", 100, 5000, 1000)
        values['Departure and Arrival Time Convenience'] = st.slider("Time Convenience", 1, 5, 4)
        values['Ease of Online Booking'] = st.slider("Online Booking", 1, 5, 4)
        values['Check-in Service'] = st.slider("Check-in", 1, 5, 4)
        values['Online Boarding'] = st.slider("Boarding", 1, 5, 4)
        values['Gate Location'] = st.slider("Gate Location", 1, 5, 4)
        values['On-board Service'] = st.slider("On-board Service", 1, 5, 4)
        values['Seat Comfort'] = st.slider("Seat Comfort", 1, 5, 4)
        values['Leg Room Service'] = st.slider("Leg Room", 1, 5, 4)
        values['Cleanliness'] = st.slider("Cleanliness", 1, 5, 4)
        values['Food and Drink'] = st.slider("Food and Drink", 1, 5, 4)
        values['In-flight Service'] = st.slider("In-flight Service", 1, 5, 4)
        values['In-flight Wifi Service'] = st.slider("Wifi", 1, 5, 3)
        values['In-flight Entertainment'] = st.slider("Entertainment", 1, 5, 4)
        values['Baggage Handling'] = st.slider("Baggage", 1, 5, 4)

        values['Gender_Male'] = 1 if st.selectbox("Geschlecht", ["Männlich", "Weiblich"]) == "Männlich" else 0
        values['Customer Type_Returning'] = 1 if st.selectbox("Kundentyp", ["Neu", "Wiederkehrend"]) == "Wiederkehrend" else 0
        values['Type of Travel_Personal'] = 1 if st.selectbox("Reiseart", ["Geschäftlich", "Privat"]) == "Privat" else 0

        klasse = st.selectbox("Klasse", ["Economy", "Economy Plus", "Business"])
        values['Class_Economy'] = 1 if klasse == "Economy" else 0
        values['Class_Economy Plus'] = 1 if klasse == "Economy Plus" else 0

        values['Delay'] = st.slider("Verspätung (Minuten)", 0, 300, 10)

        return pd.DataFrame([values])

    input_df = user_input_features()
    scaled_input = scaler.transform(input_df[feature_order])

    pred_xgb = xgb_model.predict(scaled_input)[0]
    pred_rf = rf_model.predict(scaled_input)[0]

    st.markdown("### 🔍 Modellvorhersagen:")
    st.write(f"**XGBoost:** {'Zufrieden 😊' if pred_xgb == 1 else 'Unzufrieden 😠'}")
    st.write(f"**Random Forest:** {'Zufrieden 😊' if pred_rf == 1 else 'Unzufrieden 😠'}")
