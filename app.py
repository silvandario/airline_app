import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import os

from llm import generate_action_recommendations

# Seiteneinrichtung
st.set_page_config(
    page_title="Airline Satisfaction Dashboard", 
    layout="wide", 
    page_icon="✈️", 
    initial_sidebar_state="expanded"
)

# Cache-Funktionen für Ressourcen und Daten
@st.cache_resource
def load_models():
    """Lädt alle Modelle und den Scaler"""
    models = {
        "xgb": joblib.load("models/xgb_best_model-2.pkl"),
        "rf": joblib.load("models/rf_best_model-2.pkl"),
        "lgbm": joblib.load("models/best_lgbm_optuna_classifier_2000.pkl"),
        "scaler": joblib.load("models/scaler-2.pkl")
    }
    return models

@st.cache_data
def load_shap_data_assets():
    """Lädt die vorbereiteten SHAP-Daten"""
    X_train_df = joblib.load("models/X_train_df_for_segmentation.pkl")
    shap_values_df = joblib.load("models/shap_values_df_train.pkl")
    return X_train_df, shap_values_df

@st.cache_resource
def get_shap_explainer(_model):
    """Lädt oder erstellt einen SHAP Explainer für das Modell"""
    try:
        # Versuche gespeicherten Explainer zu laden
        explainer = joblib.load("models/shap_explainer_lgbm.pkl")
    except FileNotFoundError:
        # Erstelle einen neuen Explainer, wenn keiner vorhanden ist
        explainer = shap.TreeExplainer(_model)
    return explainer

# Modelle und Daten laden
try:
    models = load_models()
    X_train_df, shap_values_df = load_shap_data_assets()
    feature_order = list(X_train_df.columns)  # Dynamische Feature-Reihenfolge aus den Trainingsdaten
    explainer = get_shap_explainer(models["lgbm"])
except Exception as e:
    st.error(f"Fehler beim Laden der Modelle oder Daten: {e}")
    feature_order = [
        'Age', 'Flight Distance', 'Departure and Arrival Time Convenience',
        'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
        'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service',
        'Cleanliness', 'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
        'In-flight Entertainment', 'Baggage Handling', 'Gender_Male',
        'Customer Type_Returning', 'Type of Travel_Personal', 'Class_Economy',
        'Class_Economy Plus', 'Delay'
    ]

# Sidebar
with st.sidebar:
    st.image("assets/logo.png", width=200)
    st.title("Airline Satisfaction Dashboard")
    st.subheader("Überblick")
    st.write(
        "Dieses Dashboard bietet Einblicke in die Zufriedenheit von Flugreisenden und ermöglicht Vorhersagen basierend auf verschiedenen Modellen."
    )
    
    # GPT-Model Selector im Sidebar
    st.markdown("---")
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
        
    st.selectbox(
        "Welches GPT-Modell verwenden?",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0 if st.session_state["openai_model"] == "gpt-3.5-turbo" else 1,
        key="openai_model"
    )

# Hintergrundbild
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

# Haupttitel
st.title("✈️ Airline Satisfaction Prediction & Insights App")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Feature Insights & Segmente", "📤 CSV Upload & Predict", "✍️ Manueller Input & Erklärung"])

# Tab 1: Feature Insights & Segmentbasierte Analyse
with tab1:
    st.header("🎯 Globale Feature Importances")
    
    # Feature Importances für alle Modelle
    models_to_display = {
        "XGBoost": (models["xgb"], "steelblue"),
        "Random Forest": (models["rf"], "darkgreen"),
        "LightGBM": (models["lgbm"], "darkorange")
    }
    
    for model_name, (model, color) in models_to_display.items():
        st.subheader(model_name)
        importance = pd.Series(model.feature_importances_, index=feature_order).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        importance.plot(kind='barh', ax=ax, color=color)
        ax.set_title(f"Top Features - {model_name}")
        st.pyplot(fig)
    
    st.markdown("---")
    
    # SHAP-basierte Segmentanalyse
    st.header("🔍 SHAP-basierte Segmentanalyse")
    st.write("""
    Diese Analyse zeigt, welche Faktoren die Zufriedenheit für spezifische Kundensegmente am stärksten beeinflussen.
    Nutzen Sie die Filter unten, um ein bestimmtes Segment zu definieren.
    """)
    
    # Segment-Filter direkt unter SHAP-basierte Segmentanalyse
    st.subheader("🎯 Segmentfilter")
    
    # Erstellen von zwei Spalten für übersichtlichere Filter-Anordnung
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        segment_class = st.selectbox(
            "Flugklasse", 
            ["Alle", "Economy", "Economy Plus", "Business"],
            key="filter_class"
        )
        
        segment_age_range = st.slider(
            "Altersbereich", 
            min_value=18, 
            max_value=85, 
            value=(25, 60),
            key="filter_age"
        )
        
        segment_travel_type = st.selectbox(
            "Reisetyp", 
            ["Alle", "Geschäftlich", "Privat"],
            key="filter_travel"
        )
    
    with filter_col2:
        segment_customer_type = st.selectbox(
            "Kundentyp", 
            ["Alle", "Neu", "Wiederkehrend"],
            key="filter_customer"
        )
        
        segment_gender = st.selectbox(
            "Geschlecht", 
            ["Alle", "Männlich", "Weiblich"],
            key="filter_gender"
        )
        
        segment_distance_range = st.slider(
            "Flugdistanz (km)", 
            min_value=0, 
            max_value=5000, 
            value=(0, 5000),
            key="filter_distance"
        )
    
    # Filtere die Daten basierend auf den Segment-Filtern
    segment_mask = pd.Series(True, index=X_train_df.index)
    
    # Klasse filtern
    if segment_class != "Alle":
        if segment_class == "Economy":
            segment_mask &= (X_train_df['Class_Economy'] == 1)
        elif segment_class == "Economy Plus":
            segment_mask &= (X_train_df['Class_Economy Plus'] == 1)
        else:  # Business
            segment_mask &= ((X_train_df['Class_Economy'] == 0) & (X_train_df['Class_Economy Plus'] == 0))
    
    # Alter filtern
    segment_mask &= (X_train_df['Age'] >= segment_age_range[0]) & (X_train_df['Age'] <= segment_age_range[1])
    
    # Reisetyp filtern
    if segment_travel_type != "Alle":
        is_personal = segment_travel_type == "Privat"
        segment_mask &= (X_train_df['Type of Travel_Personal'] == int(is_personal))
    
    # Kundentyp filtern
    if segment_customer_type != "Alle":
        is_returning = segment_customer_type == "Wiederkehrend"
        segment_mask &= (X_train_df['Customer Type_Returning'] == int(is_returning))
    
    # Geschlecht filtern
    if segment_gender != "Alle":
        is_male = segment_gender == "Männlich"
        segment_mask &= (X_train_df['Gender_Male'] == int(is_male))
    
    # Flugdistanz filtern
    segment_mask &= (X_train_df['Flight Distance'] >= segment_distance_range[0]) & (X_train_df['Flight Distance'] <= segment_distance_range[1])
    
    # Gefilterte Daten anzeigen
    filtered_df = X_train_df[segment_mask]
    filtered_shap = shap_values_df[segment_mask]
    
    num_samples = len(filtered_df)
    st.write(f"Anzahl der Datenpunkte im ausgewählten Segment: **{num_samples}**")
    
    if num_samples > 0:
        # Berechne die durchschnittlichen SHAP-Werte für das Segment
        mean_abs_shap = filtered_shap.abs().mean().sort_values(ascending=False)
        mean_shap = filtered_shap.mean().sort_values(ascending=False)
        
        # Zeige die Top N Faktoren
        top_n = 10
        st.subheader(f"Top {top_n} Einflussfaktoren für dieses Segment")
        
        # Erstelle eine Tabelle mit den Top-Einflussfaktoren
        influence_df = pd.DataFrame({
            'Feature': mean_abs_shap.index[:top_n],
            'Einfluss-Stärke': mean_abs_shap.values[:top_n],
            'Tendenz': ["Positiv ✅" if val > 0 else "Negativ ❌" for val in mean_shap.loc[mean_abs_shap.index[:top_n]].values]
        })
        
        st.dataframe(influence_df)
        
        # SHAP Summary Plot
        st.subheader("SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Bar plot der mittleren absoluten SHAP-Werte
        sorted_idx = mean_abs_shap.index[:top_n][::-1]  # Umkehren für horizontalen Plot
        colors = ['red' if mean_shap[feat] < 0 else 'green' for feat in sorted_idx]
        
        plt.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx].values, color=colors)
        plt.yticks(range(len(sorted_idx)), sorted_idx)
        plt.xlabel("Mittlerer absoluter SHAP-Wert")
        plt.title("Einfluss der Features auf die Zufriedenheit")
        
        # Legende für die Farben
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Positiver Einfluss auf Zufriedenheit'),
            Patch(facecolor='red', label='Negativer Einfluss auf Zufriedenheit')
        ]
        plt.legend(handles=legend_elements)
        
        st.pyplot(fig)
        
        # Option für SHAP Beeswarm Plot
        if st.checkbox("Zeige SHAP Beeswarm Plot (detailliertere Ansicht)"):
            st.warning("Diese Visualisierung kann bei grossen Segmenten länger dauern.")
            
            # Wähle eine Stichprobe, wenn zu viele Datenpunkte
            max_display = min(num_samples, 500)
            sample_indices = np.random.choice(filtered_df.index, size=max_display, replace=False) if num_samples > max_display else filtered_df.index
            
            # Erstelle den SHAP Beeswarm Plot
            fig, ax = plt.subplots(figsize=(10, 12))
            shap.summary_plot(
                filtered_shap.loc[sample_indices].values, 
                filtered_df.loc[sample_indices],
                feature_names=filtered_df.columns,
                max_display=top_n,
                show=False
            )
            st.pyplot(fig)
    else:
        st.warning("Keine Daten für das ausgewählte Segment gefunden. Bitte passen Sie die Filter an.")

# Tab 2: CSV Upload & Predict
with tab2:
    st.header("📤 Daten hochladen und Vorhersagen treffen")
    uploaded_file = st.file_uploader("Lade eine CSV-Datei hoch", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            if not set(feature_order).issubset(df.columns):
                missing_cols = set(feature_order) - set(df.columns)
                st.error(f"Die CSV-Datei enthält nicht alle benötigten Spalten. Fehlende Spalten: {', '.join(missing_cols)}")
            else:
                # Skaliere die Daten und mache Vorhersagen mit allen Modellen
                df_scaled = models["scaler"].transform(df[feature_order])
                
                df["XGBoost Prediction"] = models["xgb"].predict(df_scaled)
                df["Random Forest Prediction"] = models["rf"].predict(df_scaled)
                df["LightGBM Prediction"] = models["lgbm"].predict(df_scaled)
                
                # Füge Wahrscheinlichkeiten hinzu
                df["XGBoost Prob"] = models["xgb"].predict_proba(df_scaled)[:, 1]
                df["Random Forest Prob"] = models["rf"].predict_proba(df_scaled)[:, 1]
                df["LightGBM Prob"] = models["lgbm"].predict_proba(df_scaled)[:, 1]
                
                st.success("✅ Vorhersage erfolgreich durchgeführt!")
                
                # Zeige die ersten Zeilen des DataFrames
                st.subheader("Vorhersageergebnisse")
                st.dataframe(df)
                
                # Zusammenfassung der Vorhersagen
                predictions_summary = {
                    "Modell": ["XGBoost", "Random Forest", "LightGBM"],
                    "Zufriedene Kunden": [
                        df["XGBoost Prediction"].sum(),
                        df["Random Forest Prediction"].sum(),
                        df["LightGBM Prediction"].sum()
                    ],
                    "Zufriedenheitsrate": [
                        f"{df['XGBoost Prediction'].mean() * 100:.2f}%",
                        f"{df['Random Forest Prediction'].mean() * 100:.2f}%",
                        f"{df['LightGBM Prediction'].mean() * 100:.2f}%"
                    ]
                }
                
                summary_df = pd.DataFrame(predictions_summary)
                st.subheader("Zusammenfassung")
                st.write(f"Gesamtanzahl der Kunden: {len(df)}")
                st.dataframe(summary_df)
                
                # CSV-Download-Option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Ergebnisse als CSV",
                    data=csv,
                    file_name="predictions_results.csv",
                    mime="text/csv",
                )
                
        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung der Datei: {e}")

# Tab 3: Manueller Input
with tab3:
    st.subheader("✍️ Manuelle Eingabe eines Beispiels")
    
    def user_input_features_manual():
        """Funktion für die manuelle Eingabe aller Features"""
        col1, col2 = st.columns(2)
        
        values = {}
        
        # Spalte 1 - Demographische und Flugbezogene Daten
        with col1:
            st.subheader("📋 Grunddaten")
            
            values['Age'] = st.slider("Alter", 18, 80, 35)
            values['Flight Distance'] = st.slider("Flugdistanz (km)", 100, 5000, 1000)
            values['Gender_Male'] = 1 if st.selectbox("Geschlecht", ["Männlich", "Weiblich"]) == "Männlich" else 0
            values['Customer Type_Returning'] = 1 if st.selectbox("Kundentyp", ["Neu", "Wiederkehrend"]) == "Wiederkehrend" else 0
            values['Type of Travel_Personal'] = 1 if st.selectbox("Reiseart", ["Geschäftlich", "Privat"]) == "Privat" else 0
            
            klasse = st.selectbox("Flugklasse", ["Economy", "Economy Plus", "Business"])
            values['Class_Economy'] = 1 if klasse == "Economy" else 0
            values['Class_Economy Plus'] = 1 if klasse == "Economy Plus" else 0
            
            values['Delay'] = st.slider("Verspätung (Minuten)", 0, 300, 10)
        
        # Spalte 2 - Service-bezogene Daten
        with col2:
            st.subheader("⭐ Servicebewertungen")
            
            values['Departure and Arrival Time Convenience'] = st.slider("Abflugs- und Ankunftszeit", 1, 5, 4)
            values['Ease of Online Booking'] = st.slider("Online-Buchung", 1, 5, 4)
            values['Check-in Service'] = st.slider("Check-in Service", 1, 5, 4)
            values['Online Boarding'] = st.slider("Online Boarding", 1, 5, 4)
            values['Gate Location'] = st.slider("Gate-Lage", 1, 5, 4)
            values['On-board Service'] = st.slider("Bordservice", 1, 5, 4)
            values['Seat Comfort'] = st.slider("Sitzkomfort", 1, 5, 4)
            values['Leg Room Service'] = st.slider("Beinfreiheit", 1, 5, 4)
            values['Cleanliness'] = st.slider("Sauberkeit", 1, 5, 4)
            values['Food and Drink'] = st.slider("Essen & Getränke", 1, 5, 4)
            values['In-flight Service'] = st.slider("Bordservice während des Flugs", 1, 5, 4)
            values['In-flight Wifi Service'] = st.slider("WLAN während des Flugs", 1, 5, 3)
            values['In-flight Entertainment'] = st.slider("Unterhaltung während des Flugs", 1, 5, 4)
            values['Baggage Handling'] = st.slider("Gepäckabfertigung", 1, 5, 4)
        
        return pd.DataFrame([values])
    
    # Eingabeformular anzeigen
    input_df = user_input_features_manual()
    
    # Vorhersagen
    scaled_input = models["scaler"].transform(input_df[feature_order])
    
    pred_xgb = models["xgb"].predict(scaled_input)[0]
    proba_xgb = models["xgb"].predict_proba(scaled_input)[0][1]
    
    pred_rf = models["rf"].predict(scaled_input)[0]
    proba_rf = models["rf"].predict_proba(scaled_input)[0][1]
    
    pred_lgbm = models["lgbm"].predict(scaled_input)[0]
    proba_lgbm = models["lgbm"].predict_proba(scaled_input)[0][1]
    
    # Anzeige der Vorhersagen
    st.markdown("### 🔍 Modellvorhersagen:")
    
    cols = st.columns(3)
    with cols[0]:
        st.write(f"**XGBoost:** {'Zufrieden 😊' if pred_xgb == 1 else 'Unzufrieden 😠'}")
        st.write(f"Wahrscheinlichkeit: {proba_xgb:.2%}")
    
    with cols[1]:
        st.write(f"**Random Forest:** {'Zufrieden 😊' if pred_rf == 1 else 'Unzufrieden 😠'}")
        st.write(f"Wahrscheinlichkeit: {proba_rf:.2%}")
    
    with cols[2]:
        st.write(f"**LightGBM:** {'Zufrieden 😊' if pred_lgbm == 1 else 'Unzufrieden 😠'}")
        st.write(f"Wahrscheinlichkeit: {proba_lgbm:.2%}")
    
    # Funktion zur Visualisierung der Zufriedenheitswahrscheinlichkeit
    def render_rocket_probability(prob, model_name):
        """
        Visualisiert die Zufriedenheitswahrscheinlichkeit mit einer aufsteigenden Rakete.
        """
        height = int(prob * 100)
        stratosphere_level = 50
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
        return rocket_html
    
    # Visuelle Darstellung der Wahrscheinlichkeiten
    st.markdown("### 🚀 Visuelle Darstellung der Zufriedenheitswahrscheinlichkeit")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(render_rocket_probability(proba_xgb, "XGBoost"), unsafe_allow_html=True)
        if proba_xgb >= 0.5:
            st.success(f"✅ XGBoost: Zufrieden ({proba_xgb:.2%})")
        else:
            st.warning(f"⚠️ XGBoost: Nicht zufrieden ({proba_xgb:.2%})")
    
    with col2:
        st.markdown(render_rocket_probability(proba_rf, "Random Forest"), unsafe_allow_html=True)
        if proba_rf >= 0.5:
            st.success(f"✅ Random Forest: Zufrieden ({proba_rf:.2%})")
        else:
            st.warning(f"⚠️ Random Forest: Nicht zufrieden ({proba_rf:.2%})")
    
    with col3:
        st.markdown(render_rocket_probability(proba_lgbm, "LightGBM"), unsafe_allow_html=True)
        if proba_lgbm >= 0.5:
            st.success(f"✅ LightGBM: Zufrieden ({proba_lgbm:.2%})")
        else:
            st.warning(f"⚠️ LightGBM: Nicht zufrieden ({proba_lgbm:.2%})")
    
    
    # Empfehlungen zur Verbesserung der Zufriedenheit
    st.markdown("---")
    st.subheader("💡 Empfehlungen zur Verbesserung der Zufriedenheit")
    
    # Durchschnittliche Zufriedenheitswahrscheinlichkeit
    avg_prob = (proba_xgb + proba_rf + proba_lgbm) / 3
    
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
    
    # Zusammenfassung
    st.markdown("---")
    st.markdown(f"### 📊 Gesamtbewertung")
    st.markdown(f"**Durchschnittliche Zufriedenheitswahrscheinlichkeit: {avg_prob:.2%}**")
    
    if avg_prob >= 0.7:
        st.success("✅ Der Kunde wird höchstwahrscheinlich sehr zufrieden sein!")
    elif avg_prob >= 0.5:
        st.success("✅ Der Kunde wird wahrscheinlich zufrieden sein.")
    elif avg_prob >= 0.4:
        st.info("ℹ️ Der Kunde könnte zufrieden sein, aber es besteht Verbesserungspotential.")
    else:
        st.error("❌ Der Kunde wird wahrscheinlich unzufrieden sein. Massnahmen sollten ergriffen werden!")