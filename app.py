import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from llm import generate_action_recommendations
st.set_page_config(page_title="Airline Satisfaction Dashboard", layout="wide", page_icon="🧊", initial_sidebar_state="expanded"
    )

with st.sidebar:
    st.image("assets/logo.png", width=200)
    st.title("Airline Satisfaction Dashboard")
    st.subheader("Überblick")
    st.write(
        "Dieses Dashboard bietet Einblicke in die Zufriedenheit von Flugreisenden und ermöglicht Vorhersagen basierend auf verschiedenen Modellen."
    )
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("assets/background.jpg");
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    

# Lade Modelle und Scaler
xgb_model = joblib.load("models/xgb_best_model-2.pkl")
rf_model = joblib.load("models/rf_best_model-2.pkl")
scaler = joblib.load("models/scaler-2.pkl")

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
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

st.selectbox(
    "Welches GPT-Modell verwenden?",
    options=["gpt-3.5-turbo", "gpt-4"],
    index=0 if st.session_state["openai_model"] == "gpt-3.5-turbo" else 1,
    key="openai_model"
)
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
    
    
    
    def render_rocket_probability(prob, model_name):
        """
        Visualisiert die Zufriedenheitswahrscheinlichkeit mit einer aufsteigenden Rakete.
        """
        height = int(prob * 100)
        stratosphere_level = 70
        space_level = 90

        if height >= space_level:
            bg_color = "#000033"
            atmosphere_text = "🌌 WELTRAUM"
        elif height >= stratosphere_level:
            bg_color = "#0066cc"
            atmosphere_text = "🌤️ STRATOSPHÄRE"
        else:
            bg_color = "#87CEEB"
            atmosphere_text = "☁️ TROPOSPHÄRE"

        # Rakete-Flamme-Logik
        rocket = "🚀"
        flame = "🔥" if height < space_level else ""

        # HTML-Block separat als kompletter Markdown
        rocket_html = f"""
            <div style="background-color: {bg_color}; padding: 10px; border-radius: 10px; 
                        color: white; text-align: center; min-height: 220px; position: relative;">
                <h3 style="margin-bottom: 5px;">{model_name}: {prob:.2%}</h3>
                <div style="position: absolute; top: 10px; right: 10px; font-size: 16px; 
                            background-color: rgba(255,255,255,0.2); padding: 5px; border-radius: 5px;">
                    {atmosphere_text}
                </div>
                <div style="margin-top: {max(0, 100-height)}px; font-size: 36px;">
                    {rocket}
                </div>
                <div style="font-size: 24px;">{flame}</div>
                <div style="margin-top: 10px; border-top: 2px dashed white; position: relative;">
                    <div style="position: absolute; left: 0; top: 5px; font-size: 12px;">0%</div>
                    <div style="position: absolute; left: 50%; transform: translateX(-50%); top: 5px; font-size: 12px;">50%</div>
                    <div style="position: absolute; right: 0; top: 5px; font-size: 12px;">100%</div>
                </div>
            </div>
        """
        st.markdown(rocket_html, unsafe_allow_html=True)

        # Zusätzliche textliche Einschätzung
        if height >= stratosphere_level:
            st.success(f"🎯 {model_name}: Zufrieden ({prob:.2%}) – Stratosphäre erreicht!")
        else:
            st.warning(f"⚠️ {model_name}: Nicht zufrieden ({prob:.2%}) – Rakete bleibt in der Troposphäre.")
            
    st.markdown("### 🚀 Visuelle Darstellung der Zufriedenheitswahrscheinlichkeit")
    proba_xgb = xgb_model.predict_proba(scaled_input)[0][1]  # Wahrscheinlichkeit für Klasse 1 (zufrieden)
    proba_rf = rf_model.predict_proba(scaled_input)[0][1]
    
    st.markdown("### 💡 Empfehlungen zur Verbesserung der Zufriedenheit")

    # Merkmalsauswahl für Empfehlungen
    relevant_features = {k: v for k, v in input_df.iloc[0].items() if k in [
        "Online Boarding", "Type of Travel_Personal", "In-flight Wifi Service",
        "In-flight Entertainment", "Class_Economy", "Seat Comfort", "Customer Type_Returning",
        "Leg Room Service", "On-board Service", "Flight Distance", "Cleanliness",
        "Baggage Handling", "Age", "In-flight Service", "Check-in Service",
        "Ease of Online Booking", "Departure and Arrival Time Convenience", 
        "Gate Location", "Food and Drink", "Delay"
    ]}

    with st.spinner("Generiere Empfehlungen basierend auf deinen Eingaben..."):
        try:
            recommendation = generate_action_recommendations(relevant_features)
            st.success("✅ Empfehlungen erfolgreich generiert:")
            st.markdown(recommendation)
        except Exception as e:
            st.error(f"Fehler bei der Generierung: {e}")


    # Zusammenfassung beider Modelle
    st.markdown("---")
    avg_prob = (proba_xgb + proba_rf) / 2
    st.markdown(f"### 📊 Gesamtbewertung")
    st.markdown(f"**Durchschnittliche Zufriedenheitswahrscheinlichkeit: {avg_prob:.2%}**")

    if avg_prob >= 0.5:
        st.success("✅ Der Kunde wird höchstwahrscheinlich zufrieden sein!")
    elif avg_prob >= 0.4:
        st.info("ℹ️ Der Kunde könnte zufrieden sein, aber es besteht Verbesserungspotential.")
    else:
        st.error("❌ Der Kunde wird wahrscheinlich unzufrieden sein. Maßnahmen sollten ergriffen werden!")


    col1, col2 = st.columns(2)

    with col1:
        render_rocket_probability(proba_xgb, "XGBoost")

    with col2:
        render_rocket_probability(proba_rf, "Random Forest")
