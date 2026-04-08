# Mini-Programm MPI-Software-Internship
import requests
import json
from bs4 import BeautifulSoup # type: ignore

from bs4 import XMLParsedAsHTMLWarning # type: ignore
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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

def parse_pubmed_xml(xml_data: str) -> list:
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
    
def build_dataset(papers: list[dict], intervention: str):
    return {
        "intervention":intervention,
        "papers":papers
    }

# ==================================================
# Main-Function:
# ==================================================

def data_retrieval():
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
    dataset = build_dataset(papers=papers, intervention=intervention_text)
    save_json(data=dataset, path="data/processed/papers.json") 
    
    print("[longevity_ai] Data saved successfully!")


def main():
    data_retrieval()
    
if __name__ == "__main__":
    main()