# app.py

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

# Seiteneinrichtung
st.set_page_config(
    page_title="Airline Satisfaction Dashboard",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="expanded"
)

# fgagjhbo
# Annahme: llm.py existiert und hat die Funktion generate_action_recommendations
# Falls nicht, musst du sie bereitstellen oder diesen Import auskommentieren/anpassen
from llm import generate_action_recommendations, generate_segment_recommendations_from_shap


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
explainer = None # explainer wird hier nicht mehr benötigt, da keine individuelle SHAP Plots mehr da sind


try:
    models = load_models()
    X_train_df, shap_values_df = load_shap_data_assets()
    if not X_train_df.empty:
        feature_order = list(X_train_df.columns)
    else: 
        feature_order = fallback_feature_order

except Exception as e:
    st.error(f"Fehler beim Laden der Modelle oder Daten: {e}")
    feature_order = fallback_feature_order
    models = {} 
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

    max_features_for_slider = len(feature_order) if feature_order else 20
    num_features_to_display = st.slider(
        "Anzahl der Top-Features für Diagramme auswählen:",
        min_value=3,
        max_value=max(3, min(20, max_features_for_slider)),
        value=min(max(3, min(20, max_features_for_slider)), 10 if max_features_for_slider >=10 else max_features_for_slider if max_features_for_slider >=3 else 3),
        key="top_n_slider_insights_main" 
    )

    sub_tab1_global, sub_tab1_segments, sub_tab1_bcg = st.tabs(["🎯 Globale Feature Importances", "🔍 Segmente", "💡 Prioritätsmatrix"])

    with sub_tab1_global:
        st.header("🎯 Globale Feature Importances")
        if models: 
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
                if raw_importance.sum() == 0: 
                    importance = raw_importance
                else:
                    importance = raw_importance / raw_importance.sum()
                importance = importance.sort_values(ascending=False)
                if importance.empty:
                    st.write("Keine Feature Importances für dieses Modell zu visualisieren.")
                    continue
                if (importance.max() - importance.min()) == 0:
                    norm_values = pd.Series(0.5, index=importance.index) 
                else:
                    norm_values = (importance - importance.min()) / (importance.max() - importance.min())
                colors = [color_palette(val) for val in norm_values]
                fig, ax = plt.subplots(figsize=(10, 6))
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
                    ax.set_xlim(0, 0.1) 
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) 
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
            age_min = int(X_train_df['Age'].min()) if not X_train_df.empty and 'Age' in X_train_df.columns else 18
            age_max = int(X_train_df['Age'].max()) if not X_train_df.empty and 'Age' in X_train_df.columns else 85
            segment_age_range = st.slider(
                "Altersbereich", min_value=age_min, max_value=age_max, 
                value=(age_min, age_max) if age_min < age_max else (age_min, age_min +1 if age_min < 100 else age_min),
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
                value=(dist_min, dist_max) if dist_min < dist_max else (dist_min, dist_min +1 if dist_min < 10000 else dist_min),
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
            if 'Age' in X_train_df.columns:
                segment_mask &= (X_train_df['Age'] >= segment_age_range[0]) & (X_train_df['Age'] <= segment_age_range[1])
            if 'Type of Travel_Personal' in X_train_df.columns and segment_travel_type != "Alle":
                segment_mask &= (X_train_df['Type of Travel_Personal'] == (1 if segment_travel_type == "Privat" else 0))
            if 'Customer Type_Returning' in X_train_df.columns and segment_customer_type != "Alle":
                segment_mask &= (X_train_df['Customer Type_Returning'] == (1 if segment_customer_type == "Wiederkehrend" else 0))
            if 'Gender_Male' in X_train_df.columns and segment_gender != "Alle":
                segment_mask &= (X_train_df['Gender_Male'] == (1 if segment_gender == "Männlich" else 0))
            if 'Flight Distance' in X_train_df.columns:
                segment_mask &= (X_train_df['Flight Distance'] >= segment_distance_range[0]) & (X_train_df['Flight Distance'] <= segment_distance_range[1])

            filtered_df = X_train_df[segment_mask]
            filtered_shap = shap_values_df.loc[segment_mask] 
            num_samples = len(filtered_df)
            st.write(f"Anzahl der Datenpunkte im ausgewählten Segment: **{num_samples}**")

            if num_samples > 0:
                mean_abs_shap = filtered_shap.abs().mean().sort_values(ascending=False)
                mean_shap_for_tendency = filtered_shap.mean().reindex(mean_abs_shap.index)
                current_top_n_shap = min(num_features_to_display, len(mean_abs_shap))
                st.subheader(f"Top {current_top_n_shap} Einflussfaktoren für dieses Segment")
                influence_df = pd.DataFrame({
                    'Feature': mean_abs_shap.index[:current_top_n_shap],
                    'Einfluss-Stärke': mean_abs_shap.values[:current_top_n_shap],
                    'Tendenz': ["Positiv ✅" if val > 0 else ("Negativ ❌" if val < 0 else "Neutral ➖") for val in mean_shap_for_tendency.loc[mean_abs_shap.index[:current_top_n_shap]].values]
                })
                st.dataframe(influence_df.style.format({"Einfluss-Stärke": "{:.2f}"}))
                st.subheader("SHAP Summary Plot (Bar)")
                fig_bar_shap, ax_bar_shap = plt.subplots(figsize=(10, max(6, current_top_n_shap * 0.5)))
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

                if st.checkbox("Zeige SHAP Beeswarm Plot (detailliertere Ansicht)", key="beeswarm_checkbox"):
                    st.warning("Diese Visualisierung kann bei grossen Segmenten länger dauern.")
                    max_display_beeswarm = min(num_samples, 500)
                    sample_indices = np.random.choice(filtered_df.index, size=max_display_beeswarm, replace=False) if num_samples > max_display_beeswarm else filtered_df.index
                    shap_values_for_plot = filtered_shap.loc[sample_indices, feature_order].values
                    feature_values_for_plot = filtered_df.loc[sample_indices, feature_order]
                    fig_beeswarm, ax_beeswarm = plt.subplots()
                    shap.summary_plot(
                        shap_values_for_plot,
                        feature_values_for_plot,
                        max_display=current_top_n_shap,
                        show=False,
                        plot_size=(10, max(8, current_top_n_shap * 0.6))
                    )
                    plt.tight_layout()
                    st.pyplot(fig_beeswarm)
                    plt.close(fig_beeswarm)
                
                st.markdown("---")
                st.subheader("🤖 KI-basierte Analyse und Handlungsempfehlungen für dieses Segment")
                if st.button("Segment analysieren und Empfehlungen generieren", key="analyze_segment_llm_button"):
                    if not influence_df.empty:
                        with st.spinner("Generiere Handlungsempfehlungen für das Segment basierend auf SHAP-Analyse..."):
                            try:
                                recommendations_segment = generate_segment_recommendations_from_shap(influence_df)
                                st.success("✅ Empfehlungen erfolgreich generiert:")
                                st.markdown(recommendations_segment)
                            except Exception as e:
                                st.error(f"Fehler bei der Generierung der Segment-Empfehlungen: {e}")
                                st.exception(e)
                    else:
                        st.warning("Keine Daten im Segment für die Analyse vorhanden.")
            else:
                st.warning("Keine Daten für das ausgewählte Segment gefunden. Bitte passen Sie die Filter an.")
    with sub_tab1_bcg:
        st.header("💡 BCG-Matrix zur Priorisierung")

        service_features = [
            "Seat Comfort", "Cleanliness", "Food and Drink", "In-flight Wifi Service",
            "In-flight Entertainment", "Baggage Handling", "On-board Service", 
            "In-flight Service", "Check-in Service", "Online Boarding", 
            "Ease of Online Booking", "Departure and Arrival Time Convenience", 
            "Gate Location", "Leg Room Service"
        ]
        y_train = None
        # unzufriedener Kunden geht nur falls y_train.pkl verfügbar ist
        try:
            y_train = joblib.load("models/y_train.pkl")
            if isinstance(y_train, pd.DataFrame):
                y_train = y_train.squeeze()
        except Exception:
            pass

        performance_basis = st.radio("Welche Basis soll für die Performance-Werte verwendet werden?",
            options=["Gesamtdurchschnitt", "Nur unzufriedene Kunden"],
            index=0 if y_train is None else 1,
            disabled=(y_train is None),
            key="performance_basis_choice")

        df_used = X_train_df.copy()
        if not df_used.empty and not shap_values_df.empty:
            shap_means = shap_values_df.abs().mean()
            shap_means = shap_means[service_features].sort_values(ascending=False)

            if performance_basis == "Nur unzufriedene Kunden" and y_train is not None:
                df_used = df_used[y_train == 0]

            mean_ratings = df_used[service_features].mean()

            matrix_df = pd.DataFrame({
                "Feature": service_features,
                "Wichtigkeit (SHAP)": shap_means,
                "Bewertung (1-5)": mean_ratings
            }).dropna()

            x_median = matrix_df["Bewertung (1-5)"].median()
            y_median = matrix_df["Wichtigkeit (SHAP)"].median()

            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                matrix_df["Bewertung (1-5)"],
                matrix_df["Wichtigkeit (SHAP)"],
                s=200,
                alpha=0.8,
                c=matrix_df["Wichtigkeit (SHAP)"],
                cmap="coolwarm",
                edgecolors="black"
            )

            for i, row in matrix_df.iterrows():
                ax.text(row["Bewertung (1-5)"]+0.02, row["Wichtigkeit (SHAP)"], row["Feature"], fontsize=9)

            ax.axvline(x=x_median, color="gray", linestyle="--")
            ax.axhline(y=y_median, color="gray", linestyle="--")

            ax.set_xlabel("Durchschnittliche Bewertung (1 = schlecht, 5 = sehr gut)")
            ax.set_ylabel("Wichtigkeit (mittlerer SHAP-Wert)")
            ax.set_title("Prioritätsmatrix: Servicequalität vs. Einfluss auf Zufriedenheit")

            st.pyplot(fig)

            st.markdown("""
            **Interpretation:**
            - **Oben links (hoch, schlecht bewertet)**: 🔥 _Sofortige Aufmerksamkeit nötig_
            - **Unten links (niedrig, schlecht bewertet)**: 🤔 _Low Priority_
            - **Oben rechts (hoch, gut bewertet)**: ✅ _Stärken beibehalten_
            - **Unten rechts (niedrig, gut bewertet)**: 💤 _Wenig Einfluss_
            """)
        else:
            st.warning("Nicht genügend Daten für die Matrix verfügbar.")
# Tab 2: CSV Upload & Predict
with tab_upload:
    st.header("📤 Daten hochladen und Vorhersagen treffen")
    uploaded_file = st.file_uploader("Lade eine CSV-Datei hoch", type=["csv"])
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file) 
            if not set(feature_order).issubset(df_upload.columns):
                missing_cols = set(feature_order) - set(df_upload.columns)
                st.error(f"Die CSV-Datei enthält nicht alle benötigten Spalten. Fehlende Spalten: {', '.join(missing_cols)}")
            elif models is None or "scaler" not in models or models["scaler"] is None:
                st.error("Scaler-Modell nicht geladen. Vorhersage nicht möglich.")
            else:
                df_for_scaling = df_upload[feature_order].copy()
                for col in df_for_scaling.columns:
                    if df_for_scaling[col].isnull().any():
                        st.warning(f"Spalte '{col}' in den hochgeladenen Daten enthält fehlende Werte. Diese werden mit 0 ersetzt.")
                        df_for_scaling[col] = df_for_scaling[col].fillna(0)
                df_scaled = models["scaler"].transform(df_for_scaling)
                results_df = df_upload.copy()
                for model_key in ["xgb", "rf", "lgbm"]:
                    if model_key in models and models[model_key] is not None:
                        model_instance = models[model_key]
                        pred_col_name = f"{model_key.upper()} Prediction"
                        prob_col_name = f"{model_key.upper()} Prob (Zufrieden)"
                        results_df[pred_col_name] = model_instance.predict(df_scaled)
                        results_df[prob_col_name] = model_instance.predict_proba(df_scaled)[:, 1] 
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
            st.exception(e) 

# Tab 3: Manueller Input
with tab_manual:
    st.subheader("✍️ Manuelle Eingabe eines Beispiels")

    service_features_config = [
        ('Departure and Arrival Time Convenience', "Abflugs- und Ankunftszeit", 4),
        ('Ease of Online Booking', "Online-Buchung", 4),
        ('Check-in Service', "Check-in Service", 4),
        ('Online Boarding', "Online Boarding", 4),
        ('Gate Location', "Gate-Lage", 4),
        ('On-board Service', "Bordservice (allg.)", 4), 
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
        values_input = {} 
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
        return pd.DataFrame([values_input])[feature_order]

    input_df_manual = user_input_features_manual() 

    if models is None or "scaler" not in models or models["scaler"] is None:
        st.error("Scaler-Modell nicht geladen. Manuelle Vorhersage nicht möglich.")
    else:
        scaled_input_manual = models["scaler"].transform(input_df_manual) 
        predictions_manual = {} 
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
                predictions_manual[model_key] = {"pred": 0, "proba": 0.0} 
                proba_values.append(0.0)

        st.markdown("### 🔍 Modellvorhersagen:")
        pred_cols = st.columns(len(predictions_manual) if predictions_manual else 1) 
        idx = 0
        for model_key_iter, result in predictions_manual.items(): 
            with pred_cols[idx]:
                st.write(f"**{model_key_iter.upper()}:** {'Zufrieden 😊' if result['pred'] == 1 else 'Unzufrieden 😠'}")
                st.write(f"Wahrscheinlichkeit: {result['proba']:.2%}")
            idx += 1
        
        def render_rocket_probability(prob, model_name_render): 
            height = int(prob * 100)
            stratosphere_level = 50; space_level = 90 
            bg_color = "#000033" if height >= space_level else ("#0066cc" if height >= stratosphere_level else "#87CEEB")
            atmosphere_text = "🌌 WELTRAUM" if height >= space_level else ("🌤️ STRATOSPHÄRE" if height >= stratosphere_level else "☁️ TROPOSPHÄRE")
            rocket = "🚀"; flame = "🔥" if height < space_level else "" 
            return f"""<div style="background-color:{bg_color};padding:10px;border-radius:10px;color:white;text-align:center;min-height:220px;position:relative; margin-bottom:10px;">
                        <h3 style="margin-bottom:5px;">{model_name_render}: {prob:.2%}</h3>
                        <div style="position:absolute;top:10px;right:10px;font-size:16px;background-color:rgba(255,255,255,0.2);padding:5px;border-radius:5px;">{atmosphere_text}</div>
                        <div style="margin-top:{max(0,100-height)}px;font-size:36px;">{rocket}</div><div style="font-size:24px;">{flame}</div>
                        <div style="margin-top:10px;border-top:2px dashed white;position:relative;"><div style="position:absolute;left:0;top:5px;font-size:12px;">0%</div>
                        <div style="position:absolute;left:50%;transform:translateX(-50%);top:5px;font-size:12px;">50%</div>
                        <div style="position:absolute;right:0;top:5px;font-size:12px;">100%</div></div></div>"""

        st.markdown("### 🚀 Visuelle Darstellung der Zufriedenheitswahrscheinlichkeit")
        rocket_cols_display = st.columns(len(predictions_manual) if predictions_manual else 1) 
        idx = 0
        for model_key_iter, result in predictions_manual.items(): 
            with rocket_cols_display[idx]:
                st.markdown(render_rocket_probability(result['proba'], model_key_iter.upper()), unsafe_allow_html=True)
                if result['pred'] == 1: 
                    st.success(f"✅ {model_key_iter.upper()}: Zufrieden ({result['proba']:.2%})")
                else:
                    st.warning(f"⚠️ {model_key_iter.upper()}: Eher unzufrieden ({result['proba']:.2%})")
            idx +=1
        
        st.markdown("---")
        st.subheader("💡 Empfehlungen zur Verbesserung der Zufriedenheit")
        
        # Empfehlungen nur nach Button-Klick generieren
        if st.button("Empfehlungen für diesen Kunden generieren", key="generate_manual_input_recommendations_button"):
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
        avg_prob_manual = np.mean(proba_values) if proba_values else 0.0 
        st.markdown(f"**Durchschnittliche Zufriedenheitswahrscheinlichkeit (aller Modelle): {avg_prob_manual:.2%}**")
        if avg_prob_manual >= 0.7: st.success("✅ Der Kunde wird höchstwahrscheinlich sehr zufrieden sein!")
        elif avg_prob_manual >= 0.5: st.success("✅ Der Kunde wird wahrscheinlich zufrieden sein.")
        elif avg_prob_manual >= 0.4: st.info("ℹ️ Der Kunde könnte zufrieden sein, aber es besteht Verbesserungspotential.")
        else: st.error("❌ Der Kunde wird wahrscheinlich unzufrieden sein. Massnahmen sollten ergriffen werden!")