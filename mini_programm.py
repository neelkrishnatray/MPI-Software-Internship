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

def has_abstract(papers: list[dict]) -> list[dict]:
    clean = []
    for p in papers:
        if not p["abstract"]:
            continue
        if len(p["abstract"]) < 50:
            continue
        clean.append(p)
    return clean

def keyword_filter(paper: dict, keywords: dict) -> bool:
    text = (paper["title"] + " " + paper["abstract"]).lower()
    
    primary = [t.lower() for t in keywords["primary_terms"]]
    synonyms = [t.lower() for t in keywords["synonyms"]]
    mechanisms = [t.lower() for t in keywords["mechanisms"]]
    
    log = []
    log.append("\n--- PAPER: ---")
    log.append(paper["title"])
    
    # harter Filter:
    for term in primary + synonyms:
        if term in text:
            log.append(f"[MATCH primary/synonym] {term}")
            log_debug(message="\n".join(log), path="data/raw/keyword_filter.log")
            return True
    
    # weicher Filter:
    for term in mechanisms:
        if term in text:
            if any(t in text for t in ["aging", "ageing", "lifespan", "longevity", "healthspan", "age-related", "senescense"]):
                log.append(f"[MATCH mechanism] {term}")
                log_debug(message="\n".join(log), path="data/raw/keyword_filter.log")
                return True
    
    log.append("[NO MATCH]")
    log_debug(message="\n".join(log), path="data/raw/keyword_filter.log")
    return False

def add_placeholders(papers: list[dict]) -> list[dict]:
    for p in papers:
        p["study_type"] = None
        p["study_result"] = None
        p["effect_type"] = None
        p["confidence"] = None
    return papers

def validate_all(papers: list[dict], keywords: dict) -> tuple[list[dict], int]:
    results = []
    filtered_out = int(0)
    for p in papers:
        if keyword_filter(paper=p, keywords=keywords):
            results.append(p)
        else:
            filtered_out += 1
    return (results, filtered_out)
        

def classify_all(papers: list[dict]) -> list[dict]:
    results = []
    for p in papers:
        time.sleep(1)
        classified = classify_paper(paper=p)
        results.append(classified)
    return results

def assess_all(papers: list[dict]) -> list[dict]:
    results = []
    for p in papers:
        time.sleep(1)
        assessed = assess_quality(paper=p)
        results.append(assessed)
    return results

def relation_all(papers: list[dict], intervention: str) -> list[dict]:
    results = []
    for p in papers:
        time.sleep(1)
        added = add_intervention_relation(paper=p, intervention=intervention)
        results.append(added)
    return results

# ==================================================
# LLM-based generate_keywords()-function:
# ==================================================

def generate_keywords(intervention_text: str) -> dict:
    prompt = f"""
        Task: Generate search keywords for a longevity intervention.
        
        Return ONLY JSON:
        {{
            "primary_terms": [],
            "synonyms": [],
            "mechanisms": []
        }}
        
        Rules:
        - primary_terms: exact name(s) of the intervention
        - synonyms: alternative names, drug names, chemical names
        - mechanisms: biological pathways or targets

        - Be precise, avoid generic terms like "aging", "health", "therapy"
        - Keep lists short (max. 8 items per category, better if less)

        Intervention:
        {intervention_text}
    """
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    save_text(data=response.text, path=f"data/processed/keywords/{intervention_text}.json")
    
    try: 
        result = extract_json(text=response.text, error_info="generate_keywords() called")
    except:
        print("[DEBUG] generate_keywords(): extract_json() failed, handling error...")
        result = {
            "primary_terms": [],
            "synonyms": [],
            "mechanisms": []
        }

    return result
    
# ==================================================
# LLM-based classify_paper()-function:
# ==================================================

def classify_paper(paper: dict) -> dict:
    prompt = f"""
        You are a scientific reviewer.

        Task:
        Classify the "study_type" and "study_result" based on the following abstract.
        
        Return ONLY valid JSON. Do not include explanations, comments, or markdown.
        
        Schema:
        {{
            "study_type": "<one of the allowed values>",
            "study_result": "<one of the allowed values>",
            "effect_type": "<one of the allowed values>"
        }}
        
        Allowed values for "study_type":
        - 'Systematic reviews & meta-analyses'
        - 'Randomised controlled trials (RCTs)'
        - 'Observational / epidemiological studies'
        - 'Animal model studies (in vivo)'
        - 'Cell culture / in vitro studies'
        - 'In silico / computational predictions'
        - Hypothesis / perspective / commentary
        
        "study_type" definitions:
        - Systematic review: explicit methodology, multiple studies analyzed
        - RCT: randomized intervention in humans
        - Observational: cohort, case-control, epidemiology
        - Animal: experiments in animals
        - In vitro: cell culture only
        - In silico: computational only
        - Hypothesis/perspective: no original data, speculative, theoretical
        
        "study_type"-specific-rules:
        - If NO original data -> must be "Hypothesis / perspective / commentary"
        - Do NOT classify narrative or opinion papers as systematic reviews
        
        Allowed values for "study_result":
        - 'positive'
        - 'negative'
        - 'neutral'
        - 'unclear'
        
        "study_result" definitions:
        - positive: clear improvement in lifespan, healthspan, or clinically relevant function
        - negative: clear harmful or detrimental effect
        - neutral: mixed results, no significant effect, or unclear benefit
        - unclear: no measurable outcomes (e.g. hypothesis papers)
                
        "study_result"-specific-rules:
        - Molecular or mechanistic changes alone are NOT sufficient for "positive"
        - If no improvement in lifespan or function → use "neutral"
        - Hypothesis/perspective papers → "unclear"
        
        Allowed values for "effect_type":
        - lifespan
        - healthspan
        - functional
        - biomarker
        - mechanistic
        - computational
        - unclear
        
        "effect_type" definitions:
        - lifespan: Direct measurement of survival or lifespan extension
        - healthspan: Improvement in age-related disease, frailty, or overall health
        - functional: Improvement in physical or cognitive function (e.g. strength, memory)
        - biomarker: Changes in molecular or physiological markers (e.g. mTOR activity, lipids)
        - mechanistic: Cellular or pathway-level effects without clear organism-level outcome
        - computational: Predictions from models or simulations only
        - unclear: No measurable outcome or purely theoretical work
        
        "effect_type"-specific-rules:
        - Prioritize highest level of biological relevance: lifespan > healthspan > functional > biomarker > mechanistic > computational
        - If lifespan is measured -> MUST be "lifespan"
        - If abstract explicity states "no lifespan change" -> MUST NOT be "lifespan" 
        - If no organism-level outcome -> DO NOT use lifespan or healthspan
        - Molecular or pathway changes alone -> "biomarker" or "mechanistic"
        - In vitro studies -> usually "mechanistic" or "biomarker"
        - Animal studies without survival or functional outcomes -> NOT "lifespan"
        - Hypothesis / perspective papers -> MUST be "unclear"
        - If multiple effects are present -> choose the most relevant outcome
        - If unsure -> use "unclear"
        
        Rules:
        - Choose exactly one value per field
        - Do not invent categories
        - If unsure, use "unclear"
        - Use the abstract
        
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
        result = extract_json(text=response.text, error_info="classify_paper() called")
    except:
        print("[DEBUG] classify_paper(): extract_json() failed, handling error...")
        result = {
            "study_type": "unknown",
            "study_result": "unclear",
            "confidence": None
        }
    
    paper.update(result)
    return paper

# ==================================================
# LLM-based assess_quality()-function:
# ==================================================

def assess_quality(paper: dict) -> dict:
    prompt = f"""
        You are a biomedical research evaluator.
        
        Task:
        Assess the methodological quality of the following study.
        
        Return ONLY valid JSON. Do not include explanations, comments, or markdown.
        
        Schema:
        {{
            "evidence_level": "<high | moderate | low | very_low>",
            "evidence_rank": <int 1-6>,
            "study_design": "<short description>",
            "sample_size_estimate": "<small | medium | large | unknown>",
            "key_limitations": ["..."],
            "strengths": ["..."],
        }}
        
        Evidence hierarchy (STRICT):
        1 = Systematic reviews & meta-analyses (highest)
        2 = Randomised controlled trials (RCTs)
        3 = Observational / epidemiological studies
        4 = Animal model studies (in vivo)
        5 = Cell culture / in vitro studies
        6 = In silico / computational predictions (lowest)
        
        Mapping rules:
        - Rank 1-2 → evidence_level = "high"
        - Rank 3 → "moderate"
        - Rank 4 → "low"
        - Rank 5-6 → "very_low"
        
        Constraints:
        - evidence_rank MUST be consistent with the abstract
        - evidence_level MUST match the rank
        - If unclear → use rank = 6 and evidence_level = "very_low"
        - Animal or in vitro studies CANNOT be "high"
        - Prefer conservative estimates
        
        Context (may be imperfect, verify against abstract):
        - Study type: {paper.get("study_type")}
        - Study result: {paper.get("study_result")}
        - Title: {paper.get("title")}
        
        Guidelines:
        - Use abstract as primary source
        - Use context as support
        - Do not assume information not present
        - Use "unknown" if not stated
        - Keep outputs short and structured
        - Focus on methodology, not biological results
        
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
    save_text(data=response.text, path=f"data/raw/gemini/{pmid}_assessment.log")

    try:
        result = extract_json(text=response.text, error_info="assess_quality() called")
    except:
        result = {
            "quality_score": 0.0,
            "evidence_level": "very_low",
            "study_design": "unknown",
            "sample_size_estimate": "unknown",
            "key_limitations": [],
            "strengths": [],
            "confidence": 0.0
        }
        
    paper.update(result)
    return paper

# ==================================================
# LLM-based add_intervention_relation()-function:
# ==================================================

def add_intervention_relation(paper: dict, intervention: str) -> dict:
    prompt = f"""
    Task:
    Determine how the following paper relates to the longevity intervention: "{intervention}"
    
    Return ONLY valid JSON:
    {{
        "intervention_relation": "<direct | indirect | mention | unrelated>",
        "justification": "<short reason>"
    }}
    
    Definitions:
    - direct: intervention is experimentally tested or applied
    - indirect: related biological pathway/mechanism is studied
    - mention: intervention is only briefly mentioned, not central
    - unrelated: no meaningful connection
    
    Rules: 
    - Use abstract as primary source
    - Be strict: most papers are NOT direct
    - If unsure -> choose lower category ("indirect" or "mention")
    
    Title:
    {paper["title"]}
    
    Abstract:
    {paper["abstract"][:3000]}
    """
    
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    pmid = paper["pmid"]
    save_text(data=response.text, path=f"data/raw/gemini/{pmid}_relation.log")
    
    try:
        result = extract_json(text=response.text, error_info="add_intervention_relation() called")
    except:
        result = {
            "intervention_relation": "unrelated",
            "justification": "parsing_failed"
        }
        
    paper.update(result)
    return paper
    
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

def log_debug(message: str, path: str) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(message + "\n")

# ==================================================
# Main-Function:
# ==================================================

def data_retrieval(intervention_text: str) -> None:
    print("[longevity_ai] Retrieving data...")
    
    # Artikel suchen
    articles = search_pubmed(query=intervention_text)
    save_json(data=articles, path="data/raw/esearch.json")
    # articles = load_json(path="data/raw/esearch.json")
    
    # IDs extrahieren
    ids = articles["esearchresult"]["idlist"]
    
    # Details holen
    details = fetch_details(ids=ids)
    save_json(data=details, path="data/raw/esummary.json")
    # details = load_json(path="data/raw/esummary.json")
    
    # Abstracts holen
    xml_data = fetch_abstracts(ids=ids)
    save_text(data=xml_data, path="data/raw/efetch.xml")
    # xml_data = load_text(path="data/raw/efetch.xml")
    
    # XML in JSON parsen
    papers = parse_pubmed_xml(xml_data=xml_data)
    validated_papers = has_abstract(papers=papers)
    dataset = build_dataset(papers=validated_papers, intervention=intervention_text)
    save_json(data=dataset, path="data/processed/papers.json") 
    
    print("[longevity_ai] Data saved successfully!")

def validate_data() -> None:
    print("[longevity_ai] Validating papers...")
    
    # Datensatz laden
    dataset = load_json(path="data/processed/papers.json")
    papers = dataset["papers"]
    intervention_text = dataset["intervention"]
    
    # Keywords generieren mittels LLM
    keywords = generate_keywords(intervention_text=intervention_text)
    # keywords = load_json(path="data/processed/keywords/rapamycin longevity.json")
    validated_papers, filtered_out = validate_all(papers=papers, keywords=keywords)
    
    print(f"[longevity_ai] {filtered_out} paper(s) filtered out.")
    
    # Abspeichern
    save_json(
        data={"intervention": intervention_text, 
                "papers": validated_papers},
        path="data/processed/validated_papers.json"
    )
    
    print("[longevity_ai] Papers validated successfully!")

def classify_papers() -> None:
    print("[longevity_ai] Classifying papers...")
    
    # Datensatz laden
    dataset = load_json(path="data/processed/validated_papers.json")
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

def assess_qualities() -> None:
    print("[longevity_ai] Assessing qualities...")
    
    # Datensatz laden
    dataset = load_json(path="data/processed/classified_papers.json")
    papers = dataset["papers"]
    intervention_text = dataset["intervention"]
    
    # Qualität beurteilen mittels LLM
    assessed_papers = assess_all(papers=papers)
    
    # Abspeichern
    save_json(
        data={"intervention": intervention_text,
              "papers": assessed_papers},
        path="data/processed/assessed_papers.json"
    )
    
    print("[longevity_ai] Qualities assessed successfully!")
    
def add_relations() -> None:
    print("[longevity_ai] Adding relations...")
    
    # Datensatz laden
    dataset = load_json(path="data/processed/assessed_papers.json")
    papers = dataset["papers"]
    intervention_text = dataset["intervention"]
    
    # Intervention-Relation beurteilen mittels LLM
    relation_papers = relation_all(papers=papers, intervention=intervention_text)
    
    # Abspeichern
    save_json(
        data={"intervention": intervention_text,
              "papers": relation_papers},
        path="data/processed/assigned_papers.json"
    )
    
    print("[longevity_ai] Relations added successfully!")

def main(intervention: str):
    # data_retrieval(intervention)
    # validate_data()
    # classify_papers()
    # assess_qualities()
    add_relations()
    
if __name__ == "__main__":
    ageing_intervention = input("What ageing intervention would you like to research?: ")
    main(ageing_intervention)