# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns 
import lightgbm as lgb 
import shap
import os
from prompts.managementPrompts import prompt_KIBasierteEmpfehlungen_Management, prompt_SHAP_Management
from prompts.analystPrompts import prompt_SHAP_Analyst


# Seiteneinrichtung
st.set_page_config(
    page_title="Airline Satisfaction Dashboard",
    layout="wide",
    page_icon="✈️",
    initial_sidebar_state="expanded"
)

# Initialisiere den Ansichtsmodus im Session State, falls nicht vorhanden
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Management"  # Standardansicht ist "Management"


from llm import generate_action_recommendations, generate_segment_recommendations_from_shap, generate_global_importance_explanation


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
    st.title("Airline Dashboard") 

    st.markdown("---")
    st.subheader("Ansichtsmodus")
    mode_options = ["Management", "Data Analyst"]
    current_mode_index = mode_options.index(st.session_state.view_mode) 
    selected_mode = st.radio(
        "Dashboard-Ansicht wechseln:",
        options=mode_options,
        index=current_mode_index,
        captions=["Überblick für Management-Entscheidungen.", "Detaillierte Analysen für Datenexperten."]
    )
    if selected_mode != st.session_state.view_mode:
        st.session_state.view_mode = selected_mode
        st.rerun() 
    st.markdown("---")
    
    st.subheader("Überblick")
    st.write(
        "Dieses Dashboard bietet Einblicke in die Zufriedenheit von Flugreisenden und ermöglicht Vorhersagen basierend auf verschiedenen Modellen."
    )
    st.markdown("---")
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    st.selectbox(
        "GPT-Modell für KI-Analysen:",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0 if st.session_state["openai_model"] == "gpt-3.5-turbo" else 1,
        key="openai_model"
    )

# Dynamischer Hintergrund
if st.session_state.view_mode == "Data Analyst":
    data_analyst_bg_color  = "#333333"  # Dunkles Grau
    data_analyst_text_color = "#FFFFFF"  # Weiss
    st.markdown(
        f"""
        <style>
        /* Globale Stile für Data Analyst Modus */
        .stApp {{
            background-color: {data_analyst_bg_color} !important;
            background-image: none !important;
            color: {data_analyst_text_color} !important;
        }}
        /* Überschriften anpassen */
        h1, h2, h3, h4, h5, h6 {{
            color: {data_analyst_text_color} !important;
        }}
        /* Standard Text in Markdown und Alerts */
        .stMarkdown p, .stAlert p {{
            color: {data_analyst_text_color} !important;
        }}
        /* Labels von Widgets */
        .stRadio > label span, .stSelectbox > label, .stSlider > label, 
        .stFileUploader > label, .stCheckbox > label span {{ /* Span für Checkbox Label wichtig */
             color: {data_analyst_text_color} !important;
        }}
        /* Button Styling */
        .stButton > button {{
            border: 1px solid {data_analyst_text_color} !important;
            background-color: transparent !important;
            color: {data_analyst_text_color} !important;
        }}
        .stButton > button:hover {{
            border-color: {data_analyst_text_color} !important;
            background-color: rgba(255, 255, 255, 0.1) !important; 
            color: {data_analyst_text_color} !important;
        }}
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] button {{
            color: rgba(255, 255, 255, 0.7) !important; 
        }}
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
            color: {data_analyst_text_color} !important; 
            border-bottom-color: {data_analyst_text_color} !important;
        }}
        /* DataFrame/Table Text */
        .stDataFrame table, .stTable table {{
            color: {data_analyst_text_color} !important;
        }}
        .stDataFrame th, .stTable th {{
            color: {data_analyst_text_color} !important; 
            background-color: rgba(255,255,255,0.05) !important; /* Leichterer Header-Hintergrund */
        }}
         .stDataFrame td, .stTable td {{
            color: {data_analyst_text_color} !important; 
        }}
        /* Info/Warning/Error Boxen - Textfarbe anpassen */
        .stAlert {{ /* Generell für Alert-Boxen */
             color: {data_analyst_text_color} !important;
        }}
        /* Spezifische Anpassungen für Alert-Typen, falls nötig (optional) */
        .stAlert.st-ae, .stAlert.st-af, .stAlert.st-ag, .stAlert.st-ah {{ /* Streamlit interne Klassen für info, success, warning, error */
            /* Hier könnten spezifische Hintergrundfarben für die Boxen gesetzt werden, 
               die gut mit dem dunklen Hintergrund und weissem Text harmonieren, 
               aber das wird schnell komplex. Vorerst nur Textfarbe. */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

else:  # Management-Modus
    st.markdown(
        """
        <style>
        .stApp { color: initial; background-color: initial; background-image: initial;}
        h1, h2, h3, h4, h5, h6, .stMarkdown p, .stDataFrame table, .stTable table, .stAlert p { color: initial; }
        .stRadio > label span, .stSelectbox > label, .stSlider > label, .stFileUploader > label, .stCheckbox > label span { color: initial; }
        .stButton > button { border: initial; background-color: initial; color: initial; }
        .stButton > button:hover { border: initial; background-color: initial; color: initial; }
        .stTabs [data-baseweb="tab-list"] button { color: initial; }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] { color: initial; border-bottom-color: initial; }
        .stDataFrame th, .stTable th { color: initial; background-color: initial;}
        .stDataFrame td, .stTable td { color: initial; }
        </style>
        """, unsafe_allow_html=True)

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
        st.markdown("""<style>.stApp { background-color: #FFFFFF; background-image: none; }</style>""", unsafe_allow_html=True)


# Haupttitel
st.title("✈️ Airline Zufriedenheit: Dashboard & Analysen") 

# Tabs Definition basierend auf dem Ansichtsmodus
tab_definitions = []
if st.session_state.view_mode == "Management":
    tab_definitions = [
        ("📊 Übersicht & Kundengruppen", "insights_mgmt"), 
        ("✍️ Einzelbeispiel-Analyse", "manual_mgmt")
    ]
    tabs = st.tabs([t[0] for t in tab_definitions])
    tab_insights = tabs[0]
    tab_manual = tabs[1]
    tab_upload = None 
else: # Data Analyst Modus
    tab_definitions = [
        ("📊 Feature Insights", "insights_da"), 
        ("📤 CSV Upload & Predict", "upload_da"), 
        ("✍️ Manueller Input & Erklärung", "manual_da")
    ]
    tabs = st.tabs([t[0] for t in tab_definitions])
    tab_insights = tabs[0]
    tab_upload = tabs[1]
    tab_manual = tabs[2]


# Tab 1: Feature Insights / Übersicht & Kundengruppen
with tab_insights:
    header_title = "📊 Übersicht & Kundengruppen" if st.session_state.view_mode == "Management" else "📊 Feature Insights"
    st.header(header_title)

    max_features_for_slider = len(feature_order) if feature_order else 20
    num_features_to_display = st.slider(
        "Anzahl der Top-Faktoren für Diagramme auswählen:",
        min_value=3,
        max_value=max(3, min(20, max_features_for_slider)),
        value=min(max(3, min(20, max_features_for_slider)), 10 if max_features_for_slider >=10 else max_features_for_slider if max_features_for_slider >=3 else 3),
        key="top_n_slider_insights_main_common" 
    )

    sub_tab_titles = []
    if st.session_state.view_mode == "Management":
        sub_tab_titles = ["🎯 Wichtigste Treiber", "🔍 Kundengruppen-Analyse", "💡 Prioritätsmatrix"]
    else: 
        sub_tab_titles = ["🎯 Globale Feature Importances", "🔍 Segmente (SHAP-Analyse)", "💡 Prioritätsmatrix"]
    sub_tabs = st.tabs(sub_tab_titles)
    sub_tab1_global = sub_tabs[0]
    sub_tab1_segments = sub_tabs[1]
    sub_tab1_bcg = sub_tabs[2]


    with sub_tab1_global:
        current_top_importance_data_for_llm = None 
        model_name_for_llm_explanation = ""

        if st.session_state.view_mode == "Management":
            st.header("🎯 Wichtigste Treiber für Kundenzufriedenheit")
            st.info("Zusammengefasste Sicht basierend auf dem LightGBM-Modell.")
            
            if models and models.get("lgbm"):
                model_lgbm = models["lgbm"]
                
                if model_lgbm is not None and hasattr(model_lgbm, 'feature_importances_'):
                    raw_importance = pd.Series(model_lgbm.feature_importances_, index=feature_order)
                    importance = raw_importance / raw_importance.sum() if raw_importance.sum() != 0 else raw_importance
                    importance = importance.sort_values(ascending=False)

                    if not importance.empty:
                        current_top_n_global = min(num_features_to_display, len(importance))
                        current_top_importance_data_for_llm = importance.iloc[:current_top_n_global] 
                        model_name_for_llm_explanation = "dem LightGBM Modell" # Angepasster Name für den Prompt

                        st.markdown("---")
                        st.subheader("🤖 KI-basierte Zusammenfassung der Treiber")
                        if st.button("Diagramm-Erklärung generieren", key="explain_global_drivers_mgmt_btn"):
                            if current_top_importance_data_for_llm is not None and not current_top_importance_data_for_llm.empty:
                                with st.spinner("Erkläre Diagramm der wichtigsten Treiber..."):
                                    try:
                                        explanation = generate_global_importance_explanation(
                                            current_top_importance_data_for_llm, 
                                            model_name_for_llm_explanation, 
                                            st.session_state.view_mode
                                        )
                                        st.success("✅ Erklärung erfolgreich generiert:")
                                        st.markdown(explanation)
                                    except Exception as e:
                                        st.error(f"Fehler bei der Generierung der Diagramm-Erklärung: {e}")
                                        st.exception(e)
                            else:
                                st.warning("Keine Daten für die Erklärung verfügbar.")
                        st.markdown("---") 
                        
                        color_palette_lgbm = cm.coolwarm 
                        if (importance.max() - importance.min()) == 0: norm_values = pd.Series(0.5, index=importance.index) 
                        else: norm_values = (importance - importance.min()) / (importance.max() - importance.min())
                        colors = [color_palette_lgbm(val) for val in norm_values]
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        top_importance_plot_mgmt = importance.iloc[:current_top_n_global]
                        top_colors_plot_mgmt = colors[:current_top_n_global]
                        y_pos_mgmt = range(len(top_importance_plot_mgmt))

                        ax.barh(y_pos_mgmt, top_importance_plot_mgmt.values, color=top_colors_plot_mgmt)
                        ax.set_yticks(y_pos_mgmt); ax.set_yticklabels(top_importance_plot_mgmt.index)
                        ax.set_title(f"Top {current_top_n_global} Treiber") 
                        ax.set_xlabel("Relative Wichtigkeit")
                        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
                        if not top_importance_plot_mgmt.empty: ax.set_xlim(0, top_importance_plot_mgmt.iloc[0] * 1.1 if top_importance_plot_mgmt.iloc[0] > 0 else 0.1)
                        else: ax.set_xlim(0, 0.1) 
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                    else: 
                        st.write("Keine Feature Importances für LightGBM zu visualisieren.")
                else:
                     st.write("LightGBM-Modell nicht verfügbar oder hat keine Feature Importances.")
            else: 
                st.warning("LightGBM-Modell nicht geladen.")
        
        elif st.session_state.view_mode == "Data Analyst":
            st.header("🎯 Globale Feature Importances")
            if models: 
                models_to_display_da = { # Eigener Name für DA, um Verwechslung zu vermeiden
                    "XGBoost": (models.get("xgb"), cm.viridis),
                    "Random Forest": (models.get("rf"), cm.plasma),
                    "LightGBM": (models.get("lgbm"), cm.coolwarm)
                }
                for model_name, (model, color_palette) in models_to_display_da.items():
                    st.subheader(f"{model_name}")
                    if model is None or not hasattr(model, 'feature_importances_'):
                        st.write(f"Modell {model_name} nicht verfügbar oder hat keine Feature Importances.")
                        continue
                    raw_importance = pd.Series(model.feature_importances_, index=feature_order)
                    importance = raw_importance / raw_importance.sum() if raw_importance.sum() != 0 else raw_importance
                    importance = importance.sort_values(ascending=False)
                    if importance.empty:
                        st.write("Keine Feature Importances für dieses Modell zu visualisieren.")
                        continue
                    if (importance.max() - importance.min()) == 0: norm_values = pd.Series(0.5, index=importance.index) 
                    else: norm_values = (importance - importance.min()) / (importance.max() - importance.min())
                    colors = [color_palette(val) for val in norm_values]
                    fig, ax = plt.subplots(figsize=(10, 6))
                    current_top_n_global = min(num_features_to_display, len(importance))
                    top_importance = importance.iloc[:current_top_n_global]
                    top_colors = colors[:current_top_n_global]
                    y_pos = range(len(top_importance))
                    ax.barh(y_pos, top_importance.values, color=top_colors)
                    ax.set_yticks(y_pos); ax.set_yticklabels(top_importance.index)
                    ax.set_title(f"Top {current_top_n_global} Features - {model_name}")
                    ax.set_xlabel("Normalisierte Feature Importance")
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.1%}'.format(x)))
                    if not top_importance.empty: ax.set_xlim(0, top_importance.iloc[0] * 1.1 if top_importance.iloc[0] > 0 else 0.1)
                    else: ax.set_xlim(0, 0.1) 
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig) 
            else: 
                st.warning("Modelle nicht geladen.")
        st.markdown("---")


    with sub_tab1_segments:
        header_seg = "🔍 Kundengruppen-Analyse" if st.session_state.view_mode == "Management" else "🔍 Segmente (SHAP-Analyse)"
        st.header(header_seg)
        
        write_seg_intro_mgmt = """Diese Analyse zeigt, welche Faktoren die Zufriedenheit für spezifische Kundengruppen am stärksten beeinflussen.
        Nutzen Sie die Filter unten, um eine bestimmte Kundengruppe zu definieren."""
        write_seg_intro_da = """Diese Analyse zeigt, welche Faktoren die Zufriedenheit für spezifische Kundensegmente am stärksten beeinflussen (basierend auf SHAP).
        Nutzen Sie die Filter unten, um ein bestimmtes Segment zu definieren."""
        st.write(write_seg_intro_mgmt if st.session_state.view_mode == "Management" else write_seg_intro_da)
        
        st.subheader("🎯 Filter für Kundengruppen-Analyse") 
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            segment_class = st.selectbox("Flugklasse", ["Alle", "Economy", "Economy Plus", "Business"], key="filter_class_seg_common")
            age_min = int(X_train_df['Age'].min()) if not X_train_df.empty and 'Age' in X_train_df.columns else 18
            age_max = int(X_train_df['Age'].max()) if not X_train_df.empty and 'Age' in X_train_df.columns else 85
            segment_age_range = st.slider("Altersbereich", min_value=age_min, max_value=age_max, value=(age_min, age_max) if age_min < age_max else (age_min, age_min +1 if age_min < 100 else age_min), key="filter_age_seg_common")
            segment_travel_type = st.selectbox("Reisetyp", ["Alle", "Geschäftlich", "Privat"], key="filter_travel_seg_common")
        with filter_col2:
            segment_customer_type = st.selectbox("Kundentyp", ["Alle", "Neu", "Wiederkehrend"], key="filter_customer_seg_common")
            segment_gender = st.selectbox("Geschlecht", ["Alle", "Männlich", "Weiblich"], key="filter_gender_seg_common")
            dist_min = int(X_train_df['Flight Distance'].min()) if not X_train_df.empty and 'Flight Distance' in X_train_df.columns else 0
            dist_max = int(X_train_df['Flight Distance'].max()) if not X_train_df.empty and 'Flight Distance' in X_train_df.columns else 5000
            segment_distance_range = st.slider("Flugdistanz (km)", min_value=dist_min, max_value=dist_max, value=(dist_min, dist_max) if dist_min < dist_max else (dist_min, dist_min +1 if dist_min < 10000 else dist_min), key="filter_distance_seg_common")

        st.markdown("---") 
        st.subheader("🤖 KI-basierte Handlungsempfehlungen")
        
        # Diese Variable wird später im Code gefüllt, wenn num_samples > 0
        influence_df_for_llm = pd.DataFrame() # Initialisieren als leeres DataFrame

        if X_train_df.empty or shap_values_df.empty:
            st.warning("Trainingsdaten oder SHAP-Werte nicht geladen. Segmentanalyse nicht möglich.")
        else:
            segment_mask = pd.Series(True, index=X_train_df.index)
            # ... (Segment-Maskierungslogik bleibt gleich) ...
            if segment_class != "Alle":
                if segment_class == "Economy": segment_mask &= (X_train_df['Class_Economy'] == 1)
                elif segment_class == "Economy Plus": segment_mask &= (X_train_df['Class_Economy Plus'] == 1)
                else: segment_mask &= ((X_train_df['Class_Economy'] == 0) & (X_train_df['Class_Economy Plus'] == 0))
            if 'Age' in X_train_df.columns: segment_mask &= (X_train_df['Age'] >= segment_age_range[0]) & (X_train_df['Age'] <= segment_age_range[1])
            if 'Type of Travel_Personal' in X_train_df.columns and segment_travel_type != "Alle": segment_mask &= (X_train_df['Type of Travel_Personal'] == (1 if segment_travel_type == "Privat" else 0))
            if 'Customer Type_Returning' in X_train_df.columns and segment_customer_type != "Alle": segment_mask &= (X_train_df['Customer Type_Returning'] == (1 if segment_customer_type == "Wiederkehrend" else 0))
            if 'Gender_Male' in X_train_df.columns and segment_gender != "Alle": segment_mask &= (X_train_df['Gender_Male'] == (1 if segment_gender == "Männlich" else 0))
            if 'Flight Distance' in X_train_df.columns: segment_mask &= (X_train_df['Flight Distance'] >= segment_distance_range[0]) & (X_train_df['Flight Distance'] <= segment_distance_range[1])


            filtered_df = X_train_df[segment_mask]
            filtered_shap = shap_values_df.loc[segment_mask] 
            num_samples = len(filtered_df)
            st.write(f"Anzahl Datenpunkte: **{num_samples}**")

            if num_samples > 0:
                mean_abs_shap = filtered_shap.abs().mean().sort_values(ascending=False)
                mean_shap_for_tendency = filtered_shap.mean().reindex(mean_abs_shap.index)
                current_top_n_shap = min(num_features_to_display, len(mean_abs_shap))
                
                influence_df_for_llm = pd.DataFrame({
                    'Feature': mean_abs_shap.index[:current_top_n_shap],
                    'Einfluss-Stärke': mean_abs_shap.values[:current_top_n_shap], 
                    'Tendenz': ["Positiv ✅" if val > 0 else ("Negativ ❌" if val < 0 else "Neutral ➖") for val in mean_shap_for_tendency.loc[mean_abs_shap.index[:current_top_n_shap]].values]
                })
                
                if st.button("Analyse & Empfehlungen für Kundengruppe generieren", key="analyze_segment_llm_button_relocated"):
                    if not influence_df_for_llm.empty:
                        if st.session_state.view_mode == "Management":
                            with st.spinner("Generiere Handlungsempfehlungen..."):
                                try:
                                    recommendations_segment = generate_segment_recommendations_from_shap(influence_df_for_llm, st.session_state.view_mode, additional_prompt=prompt_SHAP_Management)
                                    st.success("✅ Empfehlungen erfolgreich generiert:")
                                    st.markdown(recommendations_segment)
                                except Exception as e:
                                    st.error(f"Fehler bei Generierung der Segment-Empfehlungen: {e}")
                                    st.exception(e)
                        else:
                            with st.spinner("Generiere Handlungsempfehlungen..."):
                                try:
                                    recommendations_segment = generate_segment_recommendations_from_shap(influence_df_for_llm, st.session_state.view_mode, additional_prompt=prompt_SHAP_Analyst)
                                    st.success("✅ Empfehlungen erfolgreich generiert:")
                                    st.markdown(recommendations_segment)
                                except Exception as e:
                                    st.error(f"Fehler bei Generierung der Segment-Empfehlungen: {e}")
                                    st.exception(e)
                    else:
                        st.warning("Keine Daten in der Kundengruppe für die Analyse vorhanden (influence_df ist leer).")
                st.markdown("---") 

                if st.session_state.view_mode == "Management":
                    st.subheader(f"Top {current_top_n_shap} Einflussfaktoren")
                    simplified_influence_df = influence_df_for_llm[['Feature', 'Tendenz']]
                    st.dataframe(simplified_influence_df)
                    st.subheader("Haupteinflussfaktoren auf Zufriedenheit")
                else: 
                    st.subheader(f"Top {current_top_n_shap} Einflussfaktoren (SHAP)")
                    st.dataframe(influence_df_for_llm.style.format({"Einfluss-Stärke": "{:.2f}"}))
                    st.subheader("SHAP Summary Plot (Bar)")

                fig_bar_shap, ax_bar_shap = plt.subplots(figsize=(10, max(6, current_top_n_shap * 0.5)))
                # ... (Rest des Bar Plot Codes bleibt gleich) ...
                sorted_idx = mean_abs_shap.index[:current_top_n_shap][::-1]
                bar_colors = ['green' if mean_shap_for_tendency[feat] > 0 else ('red' if mean_shap_for_tendency[feat] < 0 else 'grey') for feat in sorted_idx]
                ax_bar_shap.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx].values, color=bar_colors)
                ax_bar_shap.set_yticks(range(len(sorted_idx))); ax_bar_shap.set_yticklabels(sorted_idx)
                ax_bar_shap.set_xlabel("Mittlerer absoluter Einflusswert" if st.session_state.view_mode == "Management" else "Mittlerer absoluter SHAP-Wert")
                plot_title = "Einfluss der Faktoren auf Zufriedenheit" if st.session_state.view_mode == "Management" else "Einfluss der Features auf Zufriedenheit (SHAP)"
                ax_bar_shap.set_title(plot_title)
                from matplotlib.patches import Patch
                legend_elements = [ Patch(facecolor='green', label='Positiver Einfluss'), Patch(facecolor='red', label='Negativer Einfluss'), Patch(facecolor='grey', label='Neutraler Einfluss')]
                ax_bar_shap.legend(handles=legend_elements); plt.tight_layout(); st.pyplot(fig_bar_shap); plt.close(fig_bar_shap)


                if st.session_state.view_mode == "Data Analyst":
                    if st.checkbox("Zeige SHAP Beeswarm Plot", key="beeswarm_checkbox_da_seg_unique"):
                        st.warning("Visualisierung kann bei grossen Segmenten dauern.")
                        # ... (restlicher Beeswarm Code) ...
                        max_display_beeswarm = min(num_samples, 500)
                        sample_indices = np.random.choice(filtered_df.index, size=max_display_beeswarm, replace=False) if num_samples > max_display_beeswarm else filtered_df.index
                        shap_values_for_plot = filtered_shap.loc[sample_indices, feature_order].values
                        feature_values_for_plot = filtered_df.loc[sample_indices, feature_order]
                        fig_beeswarm, ax_beeswarm = plt.subplots()
                        shap.summary_plot(shap_values_for_plot, feature_values_for_plot, max_display=current_top_n_shap,show=False, plot_size=(10, max(8, current_top_n_shap * 0.6)))
                        plt.tight_layout(); st.pyplot(fig_beeswarm); plt.close(fig_beeswarm)
            else:
                st.warning("Keine Daten für die ausgewählte Kundengruppe. Filter anpassen.")


    with sub_tab1_bcg:
        st.header("💡 Prioritätsmatrix") 
        # ... (BCG Matrix Code bleibt unverändert wie in deinem letzten Code-Block) ...
        service_features = [
            "Seat Comfort", "Cleanliness", "Food and Drink", "In-flight Wifi Service",
            "In-flight Entertainment", "Baggage Handling", "On-board Service", 
            "In-flight Service", "Check-in Service", "Online Boarding", 
            "Ease of Online Booking", "Departure and Arrival Time Convenience", 
            "Gate Location", "Leg Room Service"
        ]
        y_train = None
        try:
            y_train_data = joblib.load("models/y_train.pkl") 
            if isinstance(y_train_data, pd.DataFrame): y_train = y_train_data.squeeze()
            elif isinstance(y_train_data, pd.Series): y_train = y_train_data
        except Exception:
            st.info("Zieldaten (y_train.pkl) für Unterscheidung 'unzufriedene Kunden' nicht gefunden. Option deaktiviert.")
        performance_basis = st.radio("Basis für Performance-Werte:",
            options=["Gesamtdurchschnitt", "Nur unzufriedene Kunden"],
            index=0 if y_train is None else 1, 
            disabled=(y_train is None), key="perf_basis_bcg_unique") # Eindeutiger Key
        df_used_bcg = X_train_df.copy() 
        if not df_used_bcg.empty and not shap_values_df.empty :
            valid_service_features_shap = [f for f in service_features if f in shap_values_df.columns]
            shap_means = shap_values_df[valid_service_features_shap].abs().mean() if valid_service_features_shap else pd.Series(dtype='float64')
            
            if performance_basis == "Nur unzufriedene Kunden" and y_train is not None:
                if len(y_train) == len(df_used_bcg):
                    df_used_bcg = df_used_bcg[y_train == 0]
                else:
                    st.warning("Länge von y_train und X_train_df stimmt nicht überein. Filter 'Nur unzufriedene Kunden' kann nicht angewendet werden.")
            
            valid_service_features_ratings = [f for f in service_features if f in df_used_bcg.columns]
            mean_ratings = df_used_bcg[valid_service_features_ratings].mean() if valid_service_features_ratings else pd.Series(dtype='float64')
            
            if not shap_means.empty and not mean_ratings.empty:
                common_features = list(set(shap_means.index) & set(mean_ratings.index))
                if common_features:
                    matrix_df = pd.DataFrame({
                        "Wichtigkeit (SHAP)": shap_means[common_features],
                        "Bewertung (1-5)": mean_ratings[common_features]
                    }) # Index ist bereits Feature
                    if not matrix_df.empty:
                        x_median = matrix_df["Bewertung (1-5)"].median()
                        y_median = matrix_df["Wichtigkeit (SHAP)"].median()
                        fig_bcg, ax_bcg = plt.subplots(figsize=(12, 10))
                        colors_quad = []
                        for feature_name_loop, row_data in matrix_df.iterrows(): 
                            if row_data["Wichtigkeit (SHAP)"] >= y_median and row_data["Bewertung (1-5)"] < x_median: colors_quad.append('red')
                            elif row_data["Wichtigkeit (SHAP)"] >= y_median and row_data["Bewertung (1-5)"] >= x_median: colors_quad.append('green')
                            elif row_data["Wichtigkeit (SHAP)"] < y_median and row_data["Bewertung (1-5)"] < x_median: colors_quad.append('orange')
                            else: colors_quad.append('blue')
                        ax_bcg.scatter(matrix_df["Bewertung (1-5)"], matrix_df["Wichtigkeit (SHAP)"], s=250, alpha=0.7, c=colors_quad, edgecolors="black")
                        for feature_name_loop, row_data in matrix_df.iterrows(): 
                            ax_bcg.text(row_data["Bewertung (1-5)"] + 0.03, row_data["Wichtigkeit (SHAP)"], feature_name_loop, fontsize=9)
                        ax_bcg.axvline(x=x_median, color="gray", linestyle="--", lw=1); ax_bcg.axhline(y=y_median, color="gray", linestyle="--", lw=1)
                        ax_bcg.set_xlabel("Durchschnittliche Bewertung (1 = schlecht, 5 = sehr gut)"); ax_bcg.set_ylabel("Wichtigkeit (mittlerer SHAP-Wert)")
                        ax_bcg.set_title("Prioritätsmatrix: Servicequalität vs. Einfluss auf Zufriedenheit")
                        if not matrix_df.empty: 
                            ax_bcg.set_xlim(2.5, 4.0) # X-Achse von 2.5 bis 4.0
                            ax_bcg.set_ylim(matrix_df["Wichtigkeit (SHAP)"].min() - (matrix_df["Wichtigkeit (SHAP)"].max()*0.05 if matrix_df["Wichtigkeit (SHAP)"].max() > 0 else 0.1 ), 
                                            matrix_df["Wichtigkeit (SHAP)"].max() * 1.05 if matrix_df["Wichtigkeit (SHAP)"].max() > 0 else 0.5)
                        plt.tight_layout(); st.pyplot(fig_bcg); plt.close(fig_bcg)
                        st.markdown("""**Interpretation:**\n- Oben links (Rot): 🔥 Sofortige Aufmerksamkeit & Investition (Wichtig & Schlecht)\n- Oben rechts (Grün): ✅ Stärken beibehalten (Wichtig & Gut)\n- Unten links (Orange): 🤔 Beobachten, geringe Priorität (Unwichtig & Schlecht)\n- Unten rechts (Blau): 💤 Kein akuter Handlungsbedarf (Unwichtig & Gut)""")
                    else: st.warning("Matrix ist leer nach Datenaufbereitung.")
                else: st.warning("Keine gemeinsamen Features für SHAP und Bewertungen gefunden.")
            else: st.warning("SHAP-Mittelwerte oder mittlere Bewertungen konnten nicht für alle Service-Features berechnet werden.")
        else: st.warning("Nicht genügend Daten für die Matrix verfügbar.")


# Tab 2: CSV Upload & Predict (Nur im Data Analyst Modus)
if st.session_state.view_mode == "Data Analyst" and tab_upload is not None:
    with tab_upload:
        st.header("📤 CSV Upload & Predict")
        # ... (Dein existierender Code für Tab 2 bleibt hier unverändert) ...
        uploaded_file = st.file_uploader("Lade eine CSV-Datei hoch", type=["csv"], key="csv_upload_da_unique_key") # Eindeutiger Key
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
                        if pd.api.types.is_numeric_dtype(df_for_scaling[col]):
                            if df_for_scaling[col].isnull().any():
                                st.warning(f"Spalte '{col}' enthält fehlende numerische Werte. Diese werden mit 0 ersetzt.")
                                df_for_scaling[col] = df_for_scaling[col].fillna(0)
                        else:
                             st.warning(f"Spalte '{col}' ist nicht numerisch und wird für die Skalierung ignoriert oder muss vorverarbeitet werden.")
                    
                    scaled_data_numpy = models["scaler"].transform(df_for_scaling) 
                    df_scaled = pd.DataFrame(scaled_data_numpy, columns=feature_order) 
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
                                "Zufriedene Kunden": int(num_satisfied),
                                "Zufriedenheitsrate": f"{satisfaction_rate_val:.2f}%"
                            })
                    if predictions_summary_list:
                        summary_df = pd.DataFrame(predictions_summary_list)
                        st.subheader("Zusammenfassung")
                        st.write(f"Gesamtanzahl der Kunden: {len(results_df)}")
                        st.dataframe(summary_df)
                    csv_output = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Ergebnisse als CSV", data=csv_output, file_name="predictions_results.csv", mime="text/csv", key="download_csv_da_unique_key") # Eindeutiger Key
            except Exception as e:
                st.error(f"Fehler bei der Verarbeitung der Datei: {e}")
                st.exception(e) 

# Tab 3: Manueller Input & Erklärung
with tab_manual:
    header_manual = "✍️ Einzelbeispiel-Analyse" if st.session_state.view_mode == "Management" else "✍️ Manueller Input & Erklärung"
    st.header(header_manual)

    # ... (Manuelle Eingabe Widgets bleiben gleich) ...
    service_features_config = [
        ('Departure and Arrival Time Convenience', "Abflugs- und Ankunftszeit", 4), ('Ease of Online Booking', "Online-Buchung", 4),
        ('Check-in Service', "Check-in Service", 4), ('Online Boarding', "Online Boarding", 4),
        ('Gate Location', "Gate-Lage", 4), ('On-board Service', "Bordservice (allg.)", 4), 
        ('Seat Comfort', "Sitzkomfort", 4), ('Leg Room Service', "Beinfreiheit", 4),
        ('Cleanliness', "Sauberkeit", 4), ('Food and Drink', "Essen & Getränke", 4),
        ('In-flight Service', "Bordservice während des Flugs", 4), ('In-flight Wifi Service', "WLAN während des Flugs", 3),
        ('In-flight Entertainment', "Unterhaltung während des Flugs", 4), ('Baggage Handling', "Gepäckabfertigung", 4)
    ]
    def user_input_features_manual():
        col1, col2 = st.columns(2)
        values_input = {} 
        with col1:
            st.subheader("📋 Grunddaten des Fluggastes")
            values_input['Age'] = st.slider("Alter", 18, 80, 35, key="manual_age_slider_cmn_key")
            values_input['Flight Distance'] = st.slider("Flugdistanz (km)", 100, 5000, 1000, key="manual_flightdist_slider_cmn_key")
            values_input['Gender_Male'] = 1 if st.selectbox("Geschlecht", ["Männlich", "Weiblich"], key="manual_gender_select_cmn_key") == "Männlich" else 0
            values_input['Customer Type_Returning'] = 1 if st.selectbox("Kundentyp", ["Neu", "Wiederkehrend"], key="manual_custtype_select_cmn_key") == "Wiederkehrend" else 0
            values_input['Type of Travel_Personal'] = 1 if st.selectbox("Reiseart", ["Geschäftlich", "Privat"], key="manual_traveltype_select_cmn_key") == "Privat" else 0
            klasse = st.selectbox("Flugklasse", ["Economy", "Economy Plus", "Business"], key="manual_class_select_cmn_key")
            values_input['Class_Economy'] = 1 if klasse == "Economy" else 0
            values_input['Class_Economy Plus'] = 1 if klasse == "Economy Plus" else 0
            values_input['Delay'] = st.slider("Verspätung (Minuten)", 0, 300, 10, key="manual_delay_slider_cmn_key")
        with col2:
            st.subheader("⭐ Servicebewertungen (1=schlecht, 5=exzellent)")
            for feat_key, display_name, default_val in service_features_config:
                values_input[feat_key] = st.slider(display_name, 1, 5, default_val, key=f"manual_slider_{feat_key.replace(' ', '_')}_cmn_key")
        return pd.DataFrame([values_input])[feature_order]
    input_df_manual = user_input_features_manual() 


    if models is None or "scaler" not in models or models["scaler"] is None:
        st.error("Scaler-Modell nicht geladen. Manuelle Vorhersage nicht möglich.")
    else:
        scaled_input_manual_numpy = models["scaler"].transform(input_df_manual) 
        scaled_input_manual = pd.DataFrame(scaled_input_manual_numpy, columns=feature_order) 
        
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
                if st.session_state.view_mode == "Data Analyst":
                    st.warning(f"{model_key.upper()}-Modell nicht geladen. Keine Vorhersage möglich.")
                predictions_manual[model_key] = {"pred": 0, "proba": 0.0} 
                proba_values.append(0.0)
        avg_prob_manual = np.mean(proba_values) if proba_values else 0.0 

        # --- Reihenfolge für Management Ansicht ---
        if st.session_state.view_mode == "Management":
            st.markdown("---")
            st.markdown(f"### 📊 Gesamtbewertung") 
            st.markdown(f"**Durchschnittliche Zufriedenheitswahrscheinlichkeit: {avg_prob_manual:.2%}**") 
            if avg_prob_manual >= 0.7: st.success("✅ Dieser Fluggasttyp wird höchstwahrscheinlich sehr zufrieden sein!")
            elif avg_prob_manual >= 0.5: st.success("✅ Dieser Fluggasttyp wird wahrscheinlich zufrieden sein.")
            elif avg_prob_manual >= 0.4: st.info("ℹ️ Zufriedenheit dieses Fluggasttyps ist unsicher, Verbesserungspotential.") 
            else: st.error("❌ Dieser Fluggasttyp wird wahrscheinlich unzufrieden sein. Massnahmen prüfen!") 
            
            st.markdown("---")
            st.subheader("💡 KI-basierte Empfehlungen") 
            if st.button("Empfehlungen generieren", key="generate_manual_recommendations_mgmt_btn_key"): # Eindeutiger Key
                relevant_features_for_llm = {k: v for k, v in input_df_manual.iloc[0].to_dict().items() if k in feature_order}
                with st.spinner("Generiere Empfehlungen..."):
                    try:
                        recommendation = generate_action_recommendations(
                        user_features=relevant_features_for_llm,
                        view_mode=st.session_state.view_mode,
                        additional_prompt=prompt_KIBasierteEmpfehlungen_Management # Neu mit RF Feature Importance werten für top 10 features
)
                        st.success("✅ Empfehlungen erfolgreich generiert:")
                        st.markdown(recommendation)
                    except Exception as e:
                        st.error(f"Fehler bei Generierung der Empfehlungen: {e}"); st.exception(e)
        
        # --- Reihenfolge für Data Analyst Ansicht ---
        elif st.session_state.view_mode == "Data Analyst":
            st.markdown("---") 
            st.markdown(f"### 📊 Gesamtbewertung") 
            st.markdown(f"**Durchschnittliche Zufriedenheitswahrscheinlichkeit (aller Modelle): {avg_prob_manual:.2%}**")
            if avg_prob_manual >= 0.7: st.success("✅ Der Kunde wird höchstwahrscheinlich sehr zufrieden sein!")
            elif avg_prob_manual >= 0.5: st.success("✅ Der Kunde wird wahrscheinlich zufrieden sein.")
            elif avg_prob_manual >= 0.4: st.info("ℹ️ Der Kunde könnte zufrieden sein, aber es besteht Verbesserungspotential.")
            else: st.error("❌ Der Kunde wird wahrscheinlich unzufrieden sein. Massnahmen sollten ergriffen werden!")

            st.markdown("---")
            st.subheader("💡 KI-basierte Empfehlungen") 
            if st.button("Empfehlungen für diesen Kunden generieren", key="generate_manual_input_recommendations_da_btn_key"): # Eindeutiger Key
                relevant_features_for_llm = {k: v for k, v in input_df_manual.iloc[0].to_dict().items() if k in feature_order}
                with st.spinner("Generiere Empfehlungen basierend auf deinen Eingaben..."):
                    try:
                        recommendation = generate_action_recommendations(relevant_features_for_llm, st.session_state.view_mode)
                        st.success("✅ Empfehlungen erfolgreich generiert:")
                        st.markdown(recommendation)
                    except Exception as e:
                        st.error(f"Fehler bei der Generierung der Empfehlungen: {e}"); st.exception(e)
            
            st.markdown("---") 
            st.markdown("### 🔍 Modellvorhersagen") 
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
                            """
            
    
            st.markdown("---") 
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
