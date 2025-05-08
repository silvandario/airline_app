from openai import OpenAI
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def generate_action_recommendations(user_features: dict) -> str:
    """
    Erzeugt systematische Empfehlungen basierend auf Nutzereingaben.
    """
    prompt = """Du bist ein Service-Optimierungs-Experte für Fluggesellschaften.
Ein Kunde hat kürzlich eine Flugreise gemacht. Basierend auf den folgenden Bewertungsmerkmalen
(1 bis 5, wobei 5 das Beste ist), sollst du Empfehlungen geben, wie die Zufriedenheit
verbessert werden kann. Konzentriere dich auf die wichtigsten Faktoren laut ML-Modell:

Hier sind die Eingaben des Kunden:
"""
    for k, v in user_features.items():
        if isinstance(v, (int, float)) and 1 <= v <= 5:
            prompt += f"- {k.replace('_', ' ')}: {v}\n"

    prompt += "\nGib max. 5 umsetzbare Vorschläge für die Airline."

    messages = [
        {"role": "system", "content": "Du bist ein Airline-Zufriedenheits-Optimierer."},
        {"role": "user", "content": prompt}
    ]
    
    stream = client.chat.completions.create(
        model=st.session_state["openai_model"],
        messages=messages,
        stream=True,
    )
    return "".join(chunk.choices[0].delta.content or "" for chunk in stream)