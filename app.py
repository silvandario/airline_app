import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns # Behalte ich bei, da es im Originalimport war
import lightgbm as lgb # Behalte ich bei, da es im Originalimport war
import shap
import os

# Annahme: llm.py existiert und hat die Funktion generate_action_recommendations
# Falls nicht, musst du sie bereitstellen oder diesen Import auskommentieren/anpassen
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


# Modelle und Daten laden
# Globale Variablen für Fallback definieren
fallback_feature_order = [
    'Age', 'Flight Distance', 'Departure and Arrival Time Convenience',
    'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
    'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service',
    'Cleanliness', 'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
    'In-flight Entertainment', 'Baggage Handling', 'Gender_Male',
    'Customer Type_Returning', 'Type of Travel_Personal', 'Class_Economy',
    'Class_Economy Plus', 'Delay'
]
models = None
X_train_df = pd.DataFrame(columns=fallback_feature_order)
shap_values_df = pd.DataFrame(columns=fallback_feature_order)
feature_order = fallback_feature_order
explainer = None


try:
    models = load_models()
    X_train_df, shap_values_df = load_shap_data_assets()
    if not X_train_df.empty:
        feature_order = list(X_train_df.columns)
    else: # Sollte nicht passieren, wenn load_shap_data_assets erfolgreich ist
        feature_order = fallback_feature_order

except Exception as e:
    st.error(f"Fehler beim Laden der Modelle oder Daten: {e}")
    # Beibehaltung der Fallback-Werte
    feature_order = fallback_feature_order
    models = {} # Leeres Dict, um KeyErrors später zu vermeiden
    X_train_df = pd.DataFrame(columns=feature_order)
    shap_values_df = pd.DataFrame(columns=feature_order)


# Sidebar
with st.sidebar:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=200)
    else:
        st.warning("Logo unter assets/logo.png nicht gefunden.")
    st.title("Airline Satisfaction Dashboard")
    st.subheader("Überblick")
    st.write(
        "Dieses Dashboard bietet Einblicke in die Zufriedenheit von Flugreisenden und ermöglicht Vorhersagen basierend auf verschiedenen Modellen."
    )

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
if os.path.exists("assets/background.jpg"):
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
else:
    st.warning("Hintergrundbild unter assets/background.jpg nicht gefunden.")

# Haupttitel
st.title("✈️ Airline Satisfaction Prediction & Insights App")

# Tabs
tab_insights, tab_upload, tab_manual = st.tabs(["📊 Feature Insights", "📤 CSV Upload & Predict", "✍️ Manueller Input & Erklärung"])


# Tab 1: Feature Insights (mit Unter-Tabs)
with tab_insights:
    st.header("📊 Feature Insights")

    # Slider für die Anzahl der Top-Features
    # Stelle sicher, dass feature_order nicht leer ist, bevor len() darauf angewendet wird.
    max_features_for_slider = len(feature_order) if feature_order else 20
    num_features_to_display = st.slider(
        "Anzahl der Top-Features für Diagramme auswählen:",
        min_value=3,
        # Stellt sicher, dass max_value mindestens min_value ist
        max_value=max(3, min(20, max_features_for_slider)),
        # Stellt sicher, dass value innerhalb von min_value und max_value liegt
        value=min(max(3, min(20, max_features_for_slider)), 10 if max_features_for_slider >=10 else max_features_for_slider if max_features_for_slider >=3 else 3),
        key="top_n_slider_insights_main" # Eindeutiger Key
    )

    sub_tab1_global, sub_tab1_segments = st.tabs(["🎯 Globale Feature Importances", "🔍 Segmente"])

    with sub_tab1_global:
        st.header("🎯 Globale Feature Importances")

        if models: # Überprüfen, ob Modelle geladen wurden
            models_to_display = {
                "XGBoost": (models.get("xgb"), cm.viridis, "Viridis"),
                "Random Forest": (models.get("rf"), cm.plasma, "Plasma"),
                "LightGBM": (models.get("lgbm"), cm.coolwarm, "Blau-Rot Gradient")
            }

            for model_name, (model, color_palette, palette_name) in models_to_display.items():
                st.subheader(f"{model_name}")
                if model is None or not hasattr(model, 'feature_importances_'):
                    st.write(f"Modell {model_name} nicht verfügbar oder hat keine Feature Importances.")
                    continue

                raw_importance = pd.Series(model.feature_importances_, index=feature_order)
                if raw_importance.sum() == 0: # Verhindert Division durch Null
                    importance = raw_importance
                else:
                    importance = raw_importance / raw_importance.sum()
                importance = importance.sort_values(ascending=False)

                if importance.empty:
                    st.write("Keine Feature Importances für dieses Modell zu visualisieren.")
                    continue
                
                # Normalisiere die Werte für Farben zwischen 0 und 1
                if (importance.max() - importance.min()) == 0:
                    norm_values = pd.Series(0.5, index=importance.index) # Alle gleiche Farbe, wenn keine Varianz
                else:
                    norm_values = (importance - importance.min()) / (importance.max() - importance.min())
                
                colors = [color_palette(val) for val in norm_values]
                fig, ax = plt.subplots(figsize=(10, 6))

                # Verwende den Wert vom Slider
                current_top_n_global = min(num_features_to_display, len(importance))
                
                top_importance = importance.iloc[:current_top_n_global]
                top_colors = colors[:current_top_n_global]

                y_pos = range(len(top_importance))
                ax.barh(y_pos, top_importance.values, color=top_colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_importance.index)
                ax.set_title(f"Top {current_top_n_global} Features - {model_name} (Normalisiert)")
                ax.set_xlabel("Normalisierte Feature Importance")
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
                if not top_importance.empty:
                    ax.set_xlim(0, top_importance.iloc[0] * 1.1 if top_importance.iloc[0] > 0 else 0.1)
                else:
                    ax.set_xlim(0, 0.1) # Fallback für leere top_importance

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Wichtig, um Speicher freizugeben
        else:
            st.warning("Modelle wurden nicht geladen. Feature Importances können nicht angezeigt werden.")
        st.markdown("---")

    with sub_tab1_segments:
        st.header("🔍 SHAP-basierte Segmentanalyse")
        st.write("""
        Diese Analyse zeigt, welche Faktoren die Zufriedenheit für spezifische Kundensegmente am stärksten beeinflussen.
        Nutzen Sie die Filter unten, um ein bestimmtes Segment zu definieren.
        """)

        st.subheader("🎯 Segmentfilter")
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            segment_class = st.selectbox(
                "Flugklasse", ["Alle", "Economy", "Economy Plus", "Business"], key="filter_class"
            )
            # Dynamische Min/Max-Werte für Slider basierend auf X_train_df, falls vorhanden
            age_min = int(X_train_df['Age'].min()) if not X_train_df.empty and 'Age' in X_train_df.columns else 18
            age_max = int(X_train_df['Age'].max()) if not X_train_df.empty and 'Age' in X_train_df.columns else 85
            segment_age_range = st.slider(
                "Altersbereich", min_value=age_min, max_value=age_max, 
                value=(age_min, age_max) if age_min < age_max else (age_min, age_min +1 if age_min < 100 else age_min), # Sicherstellen, dass value[0] <= value[1]
                key="filter_age"
            )
            segment_travel_type = st.selectbox(
                "Reisetyp", ["Alle", "Geschäftlich", "Privat"], key="filter_travel"
            )
        with filter_col2:
            segment_customer_type = st.selectbox(
                "Kundentyp", ["Alle", "Neu", "Wiederkehrend"], key="filter_customer"
            )
            segment_gender = st.selectbox(
                "Geschlecht", ["Alle", "Männlich", "Weiblich"], key="filter_gender"
            )
            dist_min = int(X_train_df['Flight Distance'].min()) if not X_train_df.empty and 'Flight Distance' in X_train_df.columns else 0
            dist_max = int(X_train_df['Flight Distance'].max()) if not X_train_df.empty and 'Flight Distance' in X_train_df.columns else 5000
            segment_distance_range = st.slider(
                "Flugdistanz (km)", min_value=dist_min, max_value=dist_max, 
                value=(dist_min, dist_max) if dist_min < dist_max else (dist_min, dist_min +1 if dist_min < 10000 else dist_min), # Sicherstellen, dass value[0] <= value[1]
                key="filter_distance"
            )

        if X_train_df.empty or shap_values_df.empty:
            st.warning("Trainingsdaten oder SHAP-Werte nicht geladen. Segmentanalyse nicht möglich.")
        else:
            segment_mask = pd.Series(True, index=X_train_df.index)
            if segment_class != "Alle":
                if segment_class == "Economy": segment_mask &= (X_train_df['Class_Economy'] == 1)
                elif segment_class == "Economy Plus": segment_mask &= (X_train_df['Class_Economy Plus'] == 1)
                else: segment_mask &= ((X_train_df['Class_Economy'] == 0) & (X_train_df['Class_Economy Plus'] == 0))
            segment_mask &= (X_train_df['Age'] >= segment_age_range[0]) & (X_train_df['Age'] <= segment_age_range[1])
            if segment_travel_type != "Alle":
                segment_mask &= (X_train_df['Type of Travel_Personal'] == (1 if segment_travel_type == "Privat" else 0))
            if segment_customer_type != "Alle":
                segment_mask &= (X_train_df['Customer Type_Returning'] == (1 if segment_customer_type == "Wiederkehrend" else 0))
            if segment_gender != "Alle":
                segment_mask &= (X_train_df['Gender_Male'] == (1 if segment_gender == "Männlich" else 0))
            segment_mask &= (X_train_df['Flight Distance'] >= segment_distance_range[0]) & (X_train_df['Flight Distance'] <= segment_distance_range[1])

            filtered_df = X_train_df[segment_mask]
            filtered_shap = shap_values_df[segment_mask]
            num_samples = len(filtered_df)
            st.write(f"Anzahl der Datenpunkte im ausgewählten Segment: **{num_samples}**")

            if num_samples > 0:
                mean_abs_shap = filtered_shap.abs().mean().sort_values(ascending=False)
                # Für die Tendenz die Reihenfolge von mean_abs_shap verwenden
                mean_shap_for_tendency = filtered_shap.mean().reindex(mean_abs_shap.index)


                # Verwende den Wert vom Slider
                current_top_n_shap = min(num_features_to_display, len(mean_abs_shap))

                st.subheader(f"Top {current_top_n_shap} Einflussfaktoren für dieses Segment")
                influence_df = pd.DataFrame({
                    'Feature': mean_abs_shap.index[:current_top_n_shap],
                    'Einfluss-Stärke': mean_abs_shap.values[:current_top_n_shap],
                    'Tendenz': ["Positiv ✅" if val > 0 else ("Negativ ❌" if val < 0 else "Neutral ➖") for val in mean_shap_for_tendency.loc[mean_abs_shap.index[:current_top_n_shap]].values]
                })
                st.dataframe(influence_df)

                st.subheader("SHAP Summary Plot (Bar)")
                fig_bar_shap, ax_bar_shap = plt.subplots(figsize=(10, max(6, current_top_n_shap * 0.5))) # Dynamische Höhe
                sorted_idx = mean_abs_shap.index[:current_top_n_shap][::-1]
                bar_colors = ['green' if mean_shap_for_tendency[feat] > 0 else ('red' if mean_shap_for_tendency[feat] < 0 else 'grey') for feat in sorted_idx]
                
                ax_bar_shap.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx].values, color=bar_colors)
                ax_bar_shap.set_yticks(range(len(sorted_idx)))
                ax_bar_shap.set_yticklabels(sorted_idx)
                ax_bar_shap.set_xlabel("Mittlerer absoluter SHAP-Wert")
                ax_bar_shap.set_title("Einfluss der Features auf die Zufriedenheit")
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', label='Positiver Einfluss'),
                    Patch(facecolor='red', label='Negativer Einfluss'),
                    Patch(facecolor='grey', label='Neutraler Einfluss')
                ]
                ax_bar_shap.legend(handles=legend_elements)
                plt.tight_layout()
                st.pyplot(fig_bar_shap)
                plt.close(fig_bar_shap)

                if st.checkbox("Zeige SHAP Beeswarm Plot (detailliertere Ansicht)"):
                    st.warning("Diese Visualisierung kann bei grossen Segmenten länger dauern.")
                    max_display_beeswarm = min(num_samples, 500)
                    sample_indices = np.random.choice(filtered_df.index, size=max_display_beeswarm, replace=False) if num_samples > max_display_beeswarm else filtered_df.index
                    
                    fig_beeswarm, ax_beeswarm = plt.subplots() # Neue Figur für Beeswarm
                    shap.summary_plot(
                        filtered_shap.loc[sample_indices].reindex(columns=feature_order, fill_value=0).values, # Sicherstellen der Spaltenreihenfolge
                        filtered_df.loc[sample_indices].reindex(columns=feature_order, fill_value=0), # Sicherstellen der Spaltenreihenfolge
                        # feature_names=filtered_df.columns, # Wird von shap intern geholt
                        max_display=current_top_n_shap, # Gesteuert durch Slider
                        show=False,
                        plot_size=(10, max(8, current_top_n_shap * 0.6))
                    )
                    plt.tight_layout()
                    st.pyplot(fig_beeswarm)
                    plt.close(fig_beeswarm) # Figur schließen
            else:
                st.warning("Keine Daten für das ausgewählte Segment gefunden. Bitte passen Sie die Filter an.")

# Tab 2: CSV Upload & Predict
with tab_upload:
    st.header("📤 Daten hochladen und Vorhersagen treffen")
    uploaded_file = st.file_uploader("Lade eine CSV-Datei hoch", type=["csv"])

    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file) # Neuer Variablenname, um Konflikte zu vermeiden

            # Überprüfen, ob alle Feature-Order-Spalten in df_upload vorhanden sind
            # und ob der Scaler und die Modelle geladen sind
            if not set(feature_order).issubset(df_upload.columns):
                missing_cols = set(feature_order) - set(df_upload.columns)
                st.error(f"Die CSV-Datei enthält nicht alle benötigten Spalten. Fehlende Spalten: {', '.join(missing_cols)}")
            elif models is None or "scaler" not in models or models["scaler"] is None:
                st.error("Scaler-Modell nicht geladen. Vorhersage nicht möglich.")
            else:
                # Nur die benötigten Spalten in der richtigen Reihenfolge auswählen
                df_for_scaling = df_upload[feature_order].copy()
                
                # Einfache Imputation für fehlende Werte (z.B. mit 0) - Anpassen falls nötig!
                for col in df_for_scaling.columns:
                    if df_for_scaling[col].isnull().any():
                        st.warning(f"Spalte '{col}' in den hochgeladenen Daten enthält fehlende Werte. Diese werden mit 0 ersetzt.")
                        df_for_scaling[col] = df_for_scaling[col].fillna(0)

                df_scaled = models["scaler"].transform(df_for_scaling)
                
                # Kopie des Original-DataFrames für die Ausgabe der Ergebnisse
                results_df = df_upload.copy()

                for model_key in ["xgb", "rf", "lgbm"]:
                    if model_key in models and models[model_key] is not None:
                        model_instance = models[model_key]
                        pred_col_name = f"{model_key.upper()} Prediction"
                        prob_col_name = f"{model_key.upper()} Prob (Zufrieden)"
                        
                        results_df[pred_col_name] = model_instance.predict(df_scaled)
                        results_df[prob_col_name] = model_instance.predict_proba(df_scaled)[:, 1] # Wahrsch. für Klasse 1
                    else:
                        st.warning(f"{model_key.upper()}-Modell nicht geladen. Keine Vorhersage möglich.")


                st.success("✅ Vorhersage erfolgreich durchgeführt!")
                st.subheader("Vorhersageergebnisse")
                st.dataframe(results_df)

                predictions_summary_list = []
                for model_key in ["xgb", "rf", "lgbm"]:
                    pred_col = f"{model_key.upper()} Prediction"
                    if pred_col in results_df.columns:
                        num_satisfied = results_df[pred_col].sum()
                        total_customers = len(results_df)
                        satisfaction_rate_val = (num_satisfied / total_customers) * 100 if total_customers > 0 else 0
                        predictions_summary_list.append({
                            "Modell": model_key.upper(),
                            "Zufriedene Kunden": num_satisfied,
                            "Zufriedenheitsrate": f"{satisfaction_rate_val:.2f}%"
                        })
                
                if predictions_summary_list:
                    summary_df = pd.DataFrame(predictions_summary_list)
                    st.subheader("Zusammenfassung")
                    st.write(f"Gesamtanzahl der Kunden: {len(results_df)}")
                    st.dataframe(summary_df)

                csv_output = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Ergebnisse als CSV",
                    data=csv_output,
                    file_name="predictions_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Fehler bei der Verarbeitung der Datei: {e}")
            st.exception(e) # Zeigt den kompletten Traceback für Debugging

# Tab 3: Manueller Input
with tab_manual:
    st.subheader("✍️ Manuelle Eingabe eines Beispiels")

    # Verwende eine Schleife für die Service-Bewertungen, um Redundanz zu reduzieren
    service_features_config = [
        ('Departure and Arrival Time Convenience', "Abflugs- und Ankunftszeit", 4),
        ('Ease of Online Booking', "Online-Buchung", 4),
        ('Check-in Service', "Check-in Service", 4),
        ('Online Boarding', "Online Boarding", 4),
        ('Gate Location', "Gate-Lage", 4),
        ('On-board Service', "Bordservice (allg.)", 4), # Geändert von "Bordservice"
        ('Seat Comfort', "Sitzkomfort", 4),
        ('Leg Room Service', "Beinfreiheit", 4),
        ('Cleanliness', "Sauberkeit", 4),
        ('Food and Drink', "Essen & Getränke", 4),
        ('In-flight Service', "Bordservice während des Flugs", 4),
        ('In-flight Wifi Service', "WLAN während des Flugs", 3),
        ('In-flight Entertainment', "Unterhaltung während des Flugs", 4),
        ('Baggage Handling', "Gepäckabfertigung", 4)
    ]

    def user_input_features_manual():
        col1, col2 = st.columns(2)
        values_input = {} # Neuer Name, um Konflikte zu vermeiden

        with col1:
            st.subheader("📋 Grunddaten")
            values_input['Age'] = st.slider("Alter", 18, 80, 35, key="manual_age_slider")
            values_input['Flight Distance'] = st.slider("Flugdistanz (km)", 100, 5000, 1000, key="manual_flightdist_slider")
            values_input['Gender_Male'] = 1 if st.selectbox("Geschlecht", ["Männlich", "Weiblich"], key="manual_gender_select") == "Männlich" else 0
            values_input['Customer Type_Returning'] = 1 if st.selectbox("Kundentyp", ["Neu", "Wiederkehrend"], key="manual_custtype_select") == "Wiederkehrend" else 0
            values_input['Type of Travel_Personal'] = 1 if st.selectbox("Reiseart", ["Geschäftlich", "Privat"], key="manual_traveltype_select") == "Privat" else 0
            klasse = st.selectbox("Flugklasse", ["Economy", "Economy Plus", "Business"], key="manual_class_select")
            values_input['Class_Economy'] = 1 if klasse == "Economy" else 0
            values_input['Class_Economy Plus'] = 1 if klasse == "Economy Plus" else 0
            values_input['Delay'] = st.slider("Verspätung (Minuten)", 0, 300, 10, key="manual_delay_slider")

        with col2:
            st.subheader("⭐ Servicebewertungen (1=schlecht, 5=exzellent)")
            for feat_key, display_name, default_val in service_features_config:
                values_input[feat_key] = st.slider(display_name, 1, 5, default_val, key=f"manual_slider_{feat_key.replace(' ', '_')}")
        
        # DataFrame mit der korrekten Feature-Reihenfolge erstellen
        return pd.DataFrame([values_input])[feature_order]


    input_df_manual = user_input_features_manual() # Neuer Variablenname

    if models is None or "scaler" not in models or models["scaler"] is None:
        st.error("Scaler-Modell nicht geladen. Manuelle Vorhersage nicht möglich.")
    else:
        scaled_input_manual = models["scaler"].transform(input_df_manual) # Neuer Variablenname

        predictions_manual = {} # Neuer Variablenname
        proba_values = []

        for model_key in ["xgb", "rf", "lgbm"]:
            if model_key in models and models[model_key] is not None:
                model_instance = models[model_key]
                pred = model_instance.predict(scaled_input_manual)[0]
                proba = model_instance.predict_proba(scaled_input_manual)[0][1]
                predictions_manual[model_key] = {"pred": pred, "proba": proba}
                proba_values.append(proba)
            else:
                st.warning(f"{model_key.upper()}-Modell nicht geladen. Keine Vorhersage möglich.")
                predictions_manual[model_key] = {"pred": 0, "proba": 0.0} # Fallback
                proba_values.append(0.0)


        st.markdown("### 🔍 Modellvorhersagen:")
        pred_cols = st.columns(len(predictions_manual) if predictions_manual else 1) # Neuer Variablenname
        
        idx = 0
        for model_key_iter, result in predictions_manual.items(): # Klare Iterationsvariablen
            with pred_cols[idx]:
                st.write(f"**{model_key_iter.upper()}:** {'Zufrieden 😊' if result['pred'] == 1 else 'Unzufrieden 😠'}")
                st.write(f"Wahrscheinlichkeit: {result['proba']:.2%}")
            idx += 1
        
        def render_rocket_probability(prob, model_name_render): # Neuer Variablenname
            height = int(prob * 100)
            stratosphere_level = 50; space_level = 90 # Semikolon für gleiche Zeile ist ok
            bg_color = "#000033" if height >= space_level else ("#0066cc" if height >= stratosphere_level else "#87CEEB")
            atmosphere_text = "🌌 WELTRAUM" if height >= space_level else ("🌤️ STRATOSPHÄRE" if height >= stratosphere_level else "☁️ TROPOSPHÄRE")
            rocket = "🚀"; flame = "🔥" if height < space_level else "" # Semikolon für gleiche Zeile ist ok
            return f"""<div style="background-color:{bg_color};padding:10px;border-radius:10px;color:white;text-align:center;min-height:220px;position:relative; margin-bottom:10px;">
                        <h3 style="margin-bottom:5px;">{model_name_render}: {prob:.2%}</h3>
                        <div style="position:absolute;top:10px;right:10px;font-size:16px;background-color:rgba(255,255,255,0.2);padding:5px;border-radius:5px;">{atmosphere_text}</div>
                        <div style="margin-top:{max(0,100-height)}px;font-size:36px;">{rocket}</div><div style="font-size:24px;">{flame}</div>
                        <div style="margin-top:10px;border-top:2px dashed white;position:relative;"><div style="position:absolute;left:0;top:5px;font-size:12px;">0%</div>
                        <div style="position:absolute;left:50%;transform:translateX(-50%);top:5px;font-size:12px;">50%</div>
                        <div style="position:absolute;right:0;top:5px;font-size:12px;">100%</div></div></div>"""

        st.markdown("### 🚀 Visuelle Darstellung der Zufriedenheitswahrscheinlichkeit")
        rocket_cols_display = st.columns(len(predictions_manual) if predictions_manual else 1) # Neuer Variablenname
        
        idx = 0
        for model_key_iter, result in predictions_manual.items(): # Klare Iterationsvariablen
            with rocket_cols_display[idx]:
                st.markdown(render_rocket_probability(result['proba'], model_key_iter.upper()), unsafe_allow_html=True)
                if result['pred'] == 1: # Einfachere Bedingung
                    st.success(f"✅ {model_key_iter.upper()}: Zufrieden ({result['proba']:.2%})")
                else:
                    st.warning(f"⚠️ {model_key_iter.upper()}: Eher unzufrieden ({result['proba']:.2%})")
            idx +=1
        
        st.markdown("---")
        st.subheader("💡 Empfehlungen zur Verbesserung der Zufriedenheit")
        
        avg_prob_manual = np.mean(proba_values) if proba_values else 0.0 # Neuer Variablenname

        # Stelle sicher, dass input_df_manual die Spalte 'Delay' und andere erwartete Spalten hat
        # Dies wird durch die Erstellung in user_input_features_manual mit feature_order sichergestellt.
        relevant_features_for_llm = {k: v for k, v in input_df_manual.iloc[0].to_dict().items() if k in feature_order}

        with st.spinner("Generiere Empfehlungen basierend auf deinen Eingaben..."):
            try:
                recommendation = generate_action_recommendations(relevant_features_for_llm)
                st.success("✅ Empfehlungen erfolgreich generiert:")
                st.markdown(recommendation)
            except Exception as e:
                st.error(f"Fehler bei der Generierung der Empfehlungen: {e}")
                st.exception(e)

        st.markdown("---")
        st.markdown(f"### 📊 Gesamtbewertung")
        st.markdown(f"**Durchschnittliche Zufriedenheitswahrscheinlichkeit (aller Modelle): {avg_prob_manual:.2%}**")
        if avg_prob_manual >= 0.7: st.success("✅ Der Kunde wird höchstwahrscheinlich sehr zufrieden sein!")
        elif avg_prob_manual >= 0.5: st.success("✅ Der Kunde wird wahrscheinlich zufrieden sein.")
        elif avg_prob_manual >= 0.4: st.info("ℹ️ Der Kunde könnte zufrieden sein, aber es besteht Verbesserungspotential.")
        else: st.error("❌ Der Kunde wird wahrscheinlich unzufrieden sein. Massnahmen sollten ergriffen werden!")