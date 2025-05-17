top_features = [
    "Online Boarding: 0.2",
    "In-flight Wifi Service: 0.1",
    "Type of Travel_Personal: 0.1",
    "In-flight Entertainment: 0.09",
    "Class_Economy: 0.07",
    "Customer Type_Returning: 0.04",
    "Leg Room Service: 0.04",
    "Seat Comfort: 0.04",
    "Flight Distance: 0.03",
    "On-board Service: 0.03"
]

prompt_KIBasierteEmpfehlungen_Management = "Die folgenden Merkmale haben sich im Modell als besonders wichtig für die Vorhersage der Zufriedenheit herausgestellt:\n"
prompt_KIBasierteEmpfehlungen_Management += "\n".join([f"- {feat}" for feat in top_features])
prompt_KIBasierteEmpfehlungen_Management += "\nBitte berücksichtige diese bei deinen Empfehlungen besonders – negative Bewertungen bei diesen Aspekten haben überdurchschnittlich grossen Einfluss. Alle anderen Features kommen danach von der Wichtigkeit her gesehen."


prompt_SHAP_Management="""
Analysiere diese Hauptfaktoren.
Formuliere 3-4 prägnante, strategische Handlungsempfehlungen für das Management, 
um die Zufriedenheit speziell dieser Kundengruppe zu verbessern. 
Fokussiere auf die kritischsten Punkte (starker negativer Einfluss) und die grösste Chancen (starker positiver Einfluss).
"""