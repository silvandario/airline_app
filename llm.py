# llm.py

from openai import OpenAI
import streamlit as st
import pandas as pd # Hinzugefügt für Typ-Hinting

# Stelle sicher, dass der API-Key geladen ist. 
# In einer echten Anwendung sollte dies sicher gehandhabt werden.
# Für Streamlit Cloud z.B. über st.secrets
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error(f"OpenAI API Key konnte nicht geladen werden: {e}. Bitte stelle sicher, dass er in den Streamlit Secrets konfiguriert ist.")
    client = None # Setze client auf None, um Fehler später abzufangen

def generate_action_recommendations(user_features: dict) -> str:
    """
    Erzeugt systematische Empfehlungen basierend auf Nutzereingaben (individuelle Bewertung).
    """
    if client is None:
        return "OpenAI Client nicht initialisiert. Empfehlungen können nicht generiert werden."

    prompt = """Du bist ein Service-Optimierungs-Experte für Fluggesellschaften.
Ein Kunde hat kürzlich eine Flugreise gemacht. Basierend auf den folgenden Bewertungsmerkmalen
(1 bis 5, wobei 5 das Beste ist), sollst du Empfehlungen geben, wie die Zufriedenheit
verbessert werden kann. Konzentriere dich auf die wichtigsten Faktoren.

Hier sind die Eingaben des Kunden:
"""
    # Filtere nur relevante Bewertungsmerkmale (1-5 Skala)
    # und formatiere sie für den Prompt
    relevant_ratings = []
    for k, v in user_features.items():
        # Versuche, den Wert in eine Zahl umzuwandeln, falls es ein String ist, der eine Zahl darstellt
        try:
            val_num = float(v)
            if 1 <= val_num <= 5: # Typische Bewertungsskala
                 relevant_ratings.append(f"- {k.replace('_', ' ').replace('_Male', '').replace(' Type Returning', '').replace(' of Travel Personal','').replace(' Class Economy','').replace('Class Economy Plus','')} (Bewertung): {int(val_num)}")
        except (ValueError, TypeError):
            # Ignoriere Features, die nicht auf der 1-5 Skala sind oder nicht konvertierbar sind
            # oder behandle sie gesondert, wenn nötig (z.B. Alter, Flugdistanz etc.)
            if k in ['Age', 'Flight Distance', 'Delay']:
                 relevant_ratings.append(f"- {k.replace('_', ' ')}: {v}")
            elif 'Gender_Male' in k and v == 1:
                 relevant_ratings.append(f"- Geschlecht: Männlich")
            elif 'Gender_Male' in k and v == 0:
                 relevant_ratings.append(f"- Geschlecht: Weiblich")
            elif 'Customer Type_Returning' in k and v == 1:
                 relevant_ratings.append(f"- Kundentyp: Wiederkehrend")
            elif 'Customer Type_Returning' in k and v == 0:
                 relevant_ratings.append(f"- Kundentyp: Neu")
            elif 'Type of Travel_Personal' in k and v == 1:
                 relevant_ratings.append(f"- Reisetyp: Privat")
            elif 'Type of Travel_Personal' in k and v == 0:
                 relevant_ratings.append(f"- Reisetyp: Geschäftlich")
            elif 'Class_Economy' in k and v == 1:
                 relevant_ratings.append(f"- Klasse: Economy")
            elif 'Class_Economy Plus' in k and v == 1:
                 relevant_ratings.append(f"- Klasse: Economy Plus")
            elif ('Class_Economy' in k and v == 0) and ('Class_Economy Plus' in user_features and user_features['Class_Economy Plus'] == 0) : # Business
                 relevant_ratings.append(f"- Klasse: Business")


    if not relevant_ratings:
        prompt += "Keine spezifischen numerischen Bewertungen im Bereich 1-5 für Service-Features gefunden. Gib allgemeine Ratschläge."
    else:
        prompt += "\n".join(relevant_ratings)


    prompt += "\n\nGib max. 5 spezifische, umsetzbare Vorschläge für die Airline, um die Zufriedenheit dieses speziellen Kundentyps basierend auf den gegebenen Informationen zu verbessern. Konzentriere dich auf die Aspekte, die wahrscheinlich eine niedrige Bewertung erhalten haben oder kritisch sind."

    messages = [
        {"role": "system", "content": "Du bist ein Airline-Zufriedenheits-Optimierer und gibst konkrete Ratschläge."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        stream = client.chat.completions.create(
            model=st.session_state.get("openai_model", "gpt-3.5-turbo"), # Fallback, falls nicht in session_state
            messages=messages,
            stream=True,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    except Exception as e:
        st.error(f"OpenAI API Fehler: {e}")
        return "Fehler bei der Kommunikation mit der OpenAI API."

def generate_segment_recommendations_from_shap(segment_shap_summary: pd.DataFrame) -> str:
    """
    Erzeugt Handlungsempfehlungen basierend auf der SHAP-Analyse eines Kundensegments.
    """
    if client is None:
        return "OpenAI Client nicht initialisiert. Empfehlungen können nicht generiert werden."

    prompt = """Du bist ein Service-Optimierungs-Experte für Fluggesellschaften.
Eine SHAP-Analyse für ein bestimmtes Kundensegment hat die folgenden Top-Einflussfaktoren auf die Kundenzufriedenheit ergeben.
'Einfluss-Stärke' ist der mittlere absolute SHAP-Wert (je höher, desto wichtiger das Feature für das Segment).
'Tendenz' gibt an, ob das Feature die Zufriedenheit im Durchschnitt eher positiv oder negativ beeinflusst hat.

Hier sind die Daten des Segments:
"""
    for index, row in segment_shap_summary.iterrows():
        prompt += f"- Feature: {row['Feature'].replace('_', ' ')}\n"
        prompt += f"  Einfluss-Stärke (relativ): {row['Einfluss-Stärke']:.2f}\n" # Als Zahl für LLM
        prompt += f"  Durchschnittliche Tendenz auf Zufriedenheit: {row['Tendenz'].replace('✅','').replace('❌','').replace('➖','')}\n"

    prompt += """
Analysiere diese Faktoren.
Gib genau 5 konkrete und umsetzbare Handlungsempfehlungen für die Fluggesellschaft, 
um die Zufriedenheit speziell für dieses Kundensegment zu verbessern. 
Konzentriere dich dabei vor allem auf die Features mit starkem negativen Einfluss 
und überlege, wie positive Einflussfaktoren weiter gestärkt oder genutzt werden können.
Formuliere die Empfehlungen klar und direkt.
"""

    messages = [
        {"role": "system", "content": "Du bist ein Airline-Strategieberater, der SHAP-Daten interpretiert und daraus Handlungsempfehlungen ableitet."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        stream = client.chat.completions.create(
            model=st.session_state.get("openai_model", "gpt-3.5-turbo"), # Fallback
            messages=messages,
            stream=True,
        )
        return "".join(chunk.choices[0].delta.content or "" for chunk in stream)
    except Exception as e:
        st.error(f"OpenAI API Fehler: {e}")
        return "Fehler bei der Kommunikation mit der OpenAI API."