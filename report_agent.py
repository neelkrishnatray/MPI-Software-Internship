###
import google.generativeai as genai
import os

genai.configure(api_key = os.gentenv("GOOGLE_API_KEY"))

client = genai.GenerativeModel("#MODEL#")
###

def generate_report(data):

    prompt = f"""
    Du bist wissenschaftlicher Autor.

    Aufgabe:
    Schreibe auf Basis der gelieferten Daten einen wissenschaftlichen Bericht.

    Schreibregeln:
    1. Verwende ausschließlich formale und sachliche Sprache.
    2. Nutze neutrale, aber analytische Ausdrucksweisen.
    3. Verwende vollständige, grammatikalisch richtige Sätze. 
    4. Keine Stichpunkte.
    5. Keine Umgangssprache.
    6. Keine Ich-Form oder persönliche Meinungen.
    7. Keine unbegründeten Behauptungen.
    8. Trenne zwischen Fakten-Aufstellung und Interpretation.
    9. Schreibe den Bericht im Stil einer wissenschaftlichen Arbeit.
    10. Erfinde keine Quellen.


    Fragestellung:
    {data["question"]}

    Verifizierte Ergebnisse:
    {data["verified_facts"]}

    Confidence Score:
    {data["confidence_score"]}

    Einschränkungen:
    {data["limitations"]}

    Benutze folgende Struktur:
    # Titel

    ## Abstract
    Kurze Zusammenfassung der Methodik und der wichtigsten Ergebnisse.

    ## Einleitung
    Einordnung des Themas und Relevanz der Fragestellung.

    ## Methodik
    Beschreibung der Datengrundlage, Recherche und Bewertung.

    ## Ergebnisse
    Darstellung der verifizierten, begründeten Ergebnisse.

    ## Diskussion
    Interpretation der Ergebnisse sowie Darstellung von Grenzen der Aussagekraft.

    ## Confidence Score Bewertung
    Einordnung und Interpretation des Scores und Aussage über Verlässlichkeit des Scores.

    ## Fazit
    Beantwortung der Fragestellung in komprimierter Form.

    ## Quellen
    Liste der bereitstehenden und verwendeten Quellen.

    Formatierung:
    Die gesamte Antwort muss im Markdown Format sein.

    """

    response = client.responses.create(
        model = "#MODEL#",
        input = prompt
    )
    return response.output_text