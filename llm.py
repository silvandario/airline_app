# llm.py

from openai import OpenAI
import streamlit as st
import pandas as pd

try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    # Statt st.error direkt hier, was die App beenden könnte, wenn llm.py importiert wird,
    # geben wir None zurück und behandeln das in der App.
    # In der App wird bereits eine Fehlermeldung angezeigt, wenn client None ist.
    # st.error(f"OpenAI API Key konnte nicht geladen werden: {e}. Bitte stelle sicher, dass er in den Streamlit Secrets konfiguriert ist.")
    print(f"Fehler beim Initialisieren des OpenAI Clients: {e}") # Für lokale Logs
    client = None

def generate_action_recommendations(user_features: dict, view_mode: str, additional_prompt: str = "") -> str:
    """
    Erzeugt systematische Empfehlungen basierend auf Nutzereingaben (individuelle Bewertung),
    angepasst an den Ansichtsmodus (Management oder Data Analyst).
    """
    if client is None:
        return "OpenAI Client nicht initialisiert. Empfehlungen können nicht generiert werden. Bitte API Key überprüfen."

    base_prompt = """Ein Fluggast hat die folgenden Eingaben gemacht. 
Bitte gib basierend darauf konkrete, umsetzbare Vorschläge für die Airline, 
um die Zufriedenheit dieses speziellen Kundentyps zu verbessern.
Konzentriere dich auf die Aspekte, die kritisch erscheinen oder niedrig bewertet wurden (Bewertungsskala 1-5, 5 ist das Beste).
Berücksichtige dabei **nicht** das Geschlecht des Kunden oder die Sitzplatzklasseals Faktor für die Bewertung oder Empfehlungen.
Diese dienen nur zur Information und dürfen keine Handlungsempfehlungen beeinflussen.

Eingaben des Kunden:
"""
    relevant_ratings = []
    for k, v in user_features.items():
        try:
            val_num = float(v)
            if 1 <= val_num <= 5: 
                # Vereinfachte Darstellung der Feature-Namen für den Prompt
                feature_display_name = k.replace('_', ' ').replace(' Type Returning', ' Wiederkehrend').replace(' of Travel Personal',' Privat').replace(' Class Economy Plus',' Economy Plus').replace(' Class Economy',' Economy').replace('Male', '').replace('Gender','Geschlecht')
                if 'Geschlecht' in feature_display_name and val_num == 1 : feature_display_name = "Geschlecht: Männlich"
                elif 'Geschlecht' in feature_display_name and val_num == 0 : feature_display_name = "Geschlecht: Weiblich"
                elif 'Wiederkehrend' in feature_display_name and val_num == 1 : feature_display_name = "Kundentyp: Wiederkehrend"
                elif 'Wiederkehrend' in feature_display_name and val_num == 0 : feature_display_name = "Kundentyp: Neu"
                elif 'Privat' in feature_display_name and val_num == 1 : feature_display_name = "Reisetyp: Privat"
                elif 'Privat' in feature_display_name and val_num == 0 : feature_display_name = "Reisetyp: Geschäftlich"
                elif 'Economy Plus' in feature_display_name and val_num == 1 : feature_display_name = "Klasse: Economy Plus"
                elif 'Economy' in feature_display_name and val_num == 1 : feature_display_name = "Klasse: Economy"
                elif user_features.get('Class_Economy', 0) == 0 and user_features.get('Class_Economy Plus', 0) == 0 and k == 'Class_Economy':
                    feature_display_name = "Klasse: Business" # Nur einmal für Business Klasse anzeigen
                    relevant_ratings.append(f"- {feature_display_name.strip()}")
                    continue # Nächste Iteration, um Duplikate für Class_Economy/Plus zu vermeiden
                elif k == 'Class_Economy Plus' and user_features.get('Class_Economy', 0) == 0 and user_features.get('Class_Economy Plus', 0) == 0:
                    continue # Bereits als Business behandelt

                if 'Bewertung' not in feature_display_name : # Nur hinzufügen, wenn es nicht bereits spezialbehandelt wurde
                     relevant_ratings.append(f"- {feature_display_name.strip()} (Bewertung): {int(val_num)}")

        except (ValueError, TypeError):
            if k in ['Age', 'Flight Distance', 'Delay']:
                relevant_ratings.append(f"- {k.replace('_', ' ')}: {v}")
    
    if not relevant_ratings:
        base_prompt += "Keine spezifischen Bewertungen oder demographischen Daten für detaillierte Empfehlungen gefunden. Gib allgemeine Ratschläge zur Steigerung der Flugzufriedenheit."
    else:
        base_prompt += "\n".join(relevant_ratings)

    if view_mode == "Management":
        system_message = "Du bist ein Managementberater für Fluggesellschaften. Formuliere deine Antworten prägnant, strategisch und direkt auf den Punkt für eine Führungsebene."
        prompt_suffix = "\n\nGib genau 3-4 übergeordnete, strategische Empfehlungen für das Management, die aus diesen Kundeneingaben abgeleitet werden können. Vermeide zu technischen Details."
    else: # Data Analyst
        system_message = "Du bist ein Datenanalyst und Optimierungsexperte für Fluggesellschaften. Gib detaillierte und datengestützte Empfehlungen."
        prompt_suffix = "\n\nGib maximal 5 detaillierte und spezifische, umsetzbare Vorschläge für die Airline. Erkläre kurz, warum diese basierend auf den Kundendaten sinnvoll sind."

    final_prompt = base_prompt + prompt_suffix
    if additional_prompt:
        final_prompt += "\n\n" + additional_prompt
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": final_prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=st.session_state.get("openai_model", "gpt-3.5-turbo"),
            messages=messages,
            stream=False, # Für einfachere Handhabung hier kein Stream
        )
        return response.choices[0].message.content
    except Exception as e:
        # st.error(f"OpenAI API Fehler: {e}") # Fehlerbehandlung in app.py
        print(f"OpenAI API Fehler in generate_action_recommendations: {e}")
        return "Fehler bei der Kommunikation mit der OpenAI API. Empfehlungen konnten nicht generiert werden."


def generate_segment_recommendations_from_shap(segment_shap_summary: pd.DataFrame, view_mode: str, additional_prompt: str = "") -> str:
    """
    Erzeugt Handlungsempfehlungen basierend auf der SHAP-Analyse eines Kundensegments,
    angepasst an den Ansichtsmodus.
    """
    if client is None:
        return "OpenAI Client nicht initialisiert. Empfehlungen können nicht generiert werden. Bitte API Key überprüfen."

    prompt = """Eine Analyse der Treiber für Kundenzufriedenheit (basierend auf SHAP-Werten) für eine spezifische Kundengruppe hat folgende Top-Faktoren ergeben:
'Einfluss-Stärke' beschreibt die relative Wichtigkeit des Faktors für diese Gruppe.
'Tendenz' zeigt, ob der Faktor die Zufriedenheit im Schnitt positiv oder negativ beeinflusst hat.

Daten der Kundengruppe:
"""
    for index, row in segment_shap_summary.iterrows():
        prompt += f"- Faktor: {row['Feature'].replace('_', ' ')}\n"
        if view_mode == "Data Analyst": # Nur für DA die Stärke explizit nennen
            prompt += f"  Relative Wichtigkeit (SHAP): {row['Einfluss-Stärke']:.2f}\n"
        prompt += f"  Durchschnittliche Tendenz auf Zufriedenheit: {row['Tendenz'].replace('✅','').replace('❌','').replace('➖','').strip()}\n"

    if view_mode == "Management":
        system_message = "Du bist ein Airline-Strategieberater für das Top-Management. Konzentriere dich auf übergeordnete strategische Hebel."
        if additional_prompt:
            prompt = prompt + additional_prompt
    else: # Data Analyst
        system_message = "Du bist ein Datenanalyst und Airline-Optimierungsexperte. Gib detaillierte und fundierte Empfehlungen."
        if additional_prompt:
            prompt = prompt + additional_prompt

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=st.session_state.get("openai_model", "gpt-3.5-turbo"),
            messages=messages,
            stream=False, 
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Fehler in generate_segment_recommendations_from_shap: {e}")
        return "Fehler bei der Kommunikation mit der OpenAI API. Empfehlungen konnten nicht generiert werden."


def generate_global_importance_explanation(feature_importances_data: pd.Series, model_name_display: str, view_mode: str) -> str:
    """
    Erzeugt eine kurze Erklärung eines Feature Importance Diagramms für das Management.
    feature_importances_data: Pandas Series mit Feature als Index und Importance als Wert.
    model_name_display: Angezeigter Name des Modells oder der Zusammenfassung.
    """
    if client is None:
        return "OpenAI Client nicht initialisiert. Erklärung kann nicht generiert werden. Bitte API Key überprüfen."

    prompt_intro = f"""Du bist ein Datenexperte, der komplexe Analyseergebnisse für das Management verständlich aufbereitet.
Das folgende Diagramm zeigt die wichtigsten Treiber für die Kundenzufriedenheit, basierend auf dem {model_name_display}. 
Die Länge der Balken indiziert die Wichtigkeit des Faktors.

Top Treiber (absteigende Wichtigkeit):
"""
    feature_list_str = ""
    for feature, importance_val in feature_importances_data.head(5).items(): # Erkläre die Top 5
        feature_list_str += f"- {feature.replace('_', ' ')} (Relative Wichtigkeit: {importance_val:.2%})\n"

    prompt = prompt_intro + feature_list_str

    prompt += """
Bitte fasse die Kernaussage dieses Diagramms in 3-5 prägnanten Sätzen zusammen. 
Welche 1-2 wichtigsten Schlussfolgerungen sollte das Management hieraus ziehen?
Die Erklärung ist für eine Management-Ebene gedacht und sollte entsprechend formuliert sein (klar, handlungsorientiert, ohne tiefgehende technische Details).
"""
    messages = [
        {"role": "system", "content": "Du bist ein Kommunikationsexperte, der Datenvisualisierungen für Führungskräfte interpretiert."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model=st.session_state.get("openai_model", "gpt-3.5-turbo"),
            messages=messages,
            stream=False,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Fehler in generate_global_importance_explanation: {e}")
        return "Fehler bei der Kommunikation mit der OpenAI API. Erklärung konnte nicht generiert werden."