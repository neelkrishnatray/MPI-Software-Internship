# Mini-Programm MPI-Software-Internship

# ==================================================
# IMPORTS:
# ==================================================
import requests
import json
from bs4 import BeautifulSoup # type: ignore

from bs4 import XMLParsedAsHTMLWarning # type: ignore
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ----- GEMINI-API-KEY: -----
from dotenv import load_dotenv # type: ignore
from google import genai
import os
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
# print(api_key[:5])

# response = client.models.generate_content(
#     model="gemini-3-flash-preview",
#     contents="Erwähne eine Intervention für Langlebigkeit im Menschen."
# )

# print(response.text)
# ---------------------------
import re
import time

# ==================================================
# Semantic-Scholar-API (Retrieve-Paper-Functions):
# ==================================================

def search_semantic_scholar(query: str, limit: int = 1) -> json:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    params = {
        "query": query,
        "limit": limit, 
        "fields": "title,abstract,year,authors"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # ----- Debugging: -----:
    print("response status code: ", response.status_code)
    print("response text: ", response.text)
    # -----------------------
    
    return data

# ==================================================
# PubMed-API (Retrieve-Papers-Functions):
# ==================================================

def search_pubmed(query: str) -> json:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": 5,
        "retmode": "json"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # ----- Debugging: -----:
    # print("response status code: ", response.status_code)
    # print("response text: ", response.text)
    # -----------------------
    
    return data

def fetch_details(ids: list[str]) -> json:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "json"
    }
    
    response = requests.get(url, params=params)
    return response.json()

def fetch_abstracts(ids: list[str]) -> str:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status() # siltent-failure vermeiden
    return response.text        # XML

def parse_pubmed_xml(xml_data: str) -> list[dict]:
    soup = BeautifulSoup(xml_data, "lxml")
    
    articles = soup.find_all("pubmedarticle")
    papers = []
    
    for article in articles:
        # Abstract (kann mehrere Teile haben)
        abstract_parts = article.find_all("abstracttext")
        if not abstract_parts:
            continue
        abstract = " ".join([a.text for a in abstract_parts])
        
        # PMID
        pmid_tag = article.find("pmid")
        pmid = pmid_tag.text if pmid_tag else None
        
        # Title
        title_tag = article.find("articletitle")
        title = title_tag.text if title_tag else None
        
        # Journal
        journal_tag = article.find("title")
        journal = journal_tag.text if journal_tag else None
        
        # Year
        pubdate_tag = article.find("pubdate")
        pubdate = pubdate_tag.text if pubdate_tag else None
        
        # Authors
        authors = []
        author_list = article.find_all("author")
        for author in author_list:
            lastname = author.find("lastname")
            firstname = author.find("forename")
            
            if lastname and firstname:
                authors.append(f"{firstname.text} {lastname.text}")
                
        papers.append({
            "pmid":pmid,
            "title":title,
            "abstract":abstract,
            "journal":journal,
            "pubdate":pubdate,
            "authors":authors
        })
    
    return papers

# ==================================================
# Paper-Handling-Functions:
# ==================================================

def validate_papers(papers: list[dict]) -> list[dict]:
    clean = []
    for p in papers:
        if not p["abstract"]:
            continue
        if len(p["abstract"]) < 50:
            continue
        clean.append(p)
    return clean

def add_placeholders(papers: list[dict]) -> list[dict]:
    for p in papers:
        p["study_type"] = None
        p["study_result"] = None
        p["confidence"] = None
    return papers

def classify_paper(paper: dict) -> dict:
    prompt = f"""
        You are a scientific reviewer.

        Task:
        Classify the study described in the abstract.
        
        Return ONLY valid JSON. Do not include explanations, comments, or markdown.
        
        Schema:
        {{
            "study_type": "<one of the allowed values>",
            "study_result": "<one of the allowed values>",
            "confidence": <number between 0 and 1>
        }}
        
        Allowed values for "study_type":
        - 'Systematic reviews & meta-analyses'
        - 'Randomised controlled trials (RCTs)'
        - 'Observational / epidemiological studies'
        - 'Animal model studies (in vivo)'
        - 'Cell culture / in vitro studies'
        - 'In silico / computational predictions'
        
        Allowed values for "study_result":
        - 'positive'
        - 'negative'
        - 'neutral'
        - 'unclear'
        
        Rules:
        - Choose exactly one value per field
        - Do not invent categories
        - If unsure, use "unclear"
        - Confidence must be a float betweenn 0 and 1
        
        Abstract:
        '''
        {paper["abstract"][:3000]}
        '''
    """
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    pmid = paper["pmid"]
    save_text(data=response.text, path=f"data/raw/gemini/{pmid}_classification.log")
    
    try: 
        result = extract_json(response.text, error_info="classify_paper() called")
    except:
        print("[DEBUG] classify_paper(): extract_json() failed, handling error...")
        result = {
            "study_type": "unknown",
            "study_result": "unclear",
            "confidence": 0.0
        }
    
    paper.update(result)
    return paper

def classify_all(papers: list[dict]) -> list[dict]:
    results = []
    for p in papers:
        time.sleep(1)
        classified = classify_paper(p)
        results.append(classified)
    return results
    
    

# ==================================================
# Save & Load-Functions:
# ==================================================

def save_json(data: json, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj=data, fp=file, indent=2, ensure_ascii=False)
        
def load_json(path: str) -> json:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
    
def save_text(data: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.write(data)
        
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
    
# ==================================================
# Helper-Functions:
# ==================================================

def pretty(data: json) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
def build_dataset(papers: list[dict], intervention: str) -> dict:
    return {
        "intervention":intervention,
        "papers":papers
    }
    
def extract_json(text: str, error_info: str):
    match = re.search(pattern=r"\{.*\}", string=text, flags=re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No valid json found: {error_info}")

# ==================================================
# Main-Function:
# ==================================================

def data_retrieval() -> None:
    print("[longevity_ai] Retrieving data...")
    intervention_text = "rapamycin longevity"
    
    # Schritt 1: Artikel suchen
    articles = search_pubmed(query=intervention_text)
    save_json(data=articles, path="data/raw/esearch.json")
    # articles = load_json(path="data/raw/esearch.json")
    
    # Schritt 2: IDs extrahieren
    ids = articles["esearchresult"]["idlist"]
    
    # Schritt 3: Details holen
    details = fetch_details(ids=ids)
    save_json(data=details, path="data/raw/esummary.json")
    # details = load_json(path="data/raw/esummary.json")
    
    # Schritt 4: Abstracts holen
    xml_data = fetch_abstracts(ids=ids)
    save_text(data=xml_data, path="data/raw/efetch.xml")
    # xml_data = load_text(path="data/raw/efetch.xml")
    
    # Schritt 5: XML in JSON parsen
    papers = parse_pubmed_xml(xml_data=xml_data)
    validated_papers = validate_papers(papers=papers)
    dataset = build_dataset(papers=validated_papers, intervention=intervention_text)
    save_json(data=dataset, path="data/processed/papers.json") 
    
    print("[longevity_ai] Data saved successfully!")

def classify_papers() -> None:
    print("[longevity_ai] Classifying papers...")
    
    # Datensatz laden
    dataset = load_json(path="data/processed/papers.json")
    papers = dataset["papers"]
    intervention_text = dataset["intervention"]
    
    # Klassifizieren mittels LLM
    papers = add_placeholders(papers=papers)
    classified_papers = classify_all(papers=papers)
    
    # Abspeichern
    save_json(
        data={"intervention": intervention_text, 
              "papers": classified_papers},
        path="data/processed/classified_papers.json"
    )
    
    print("[longevity_ai] Papers classified successfully!")
    

def main():
    data_retrieval()
    classify_papers()
    
if __name__ == "__main__":
    main()