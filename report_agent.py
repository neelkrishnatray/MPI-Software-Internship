# ==================================================
# report_agent.py
# Generates the final research report
# and saves it as Markdown and PDF.
#
# Input:  data/processed/ranked_papers.json
#         data/processed/summary.json
# Output: outputs/report.md
#         outputs/report.pdf  (optional, requires LaTeX)
# ==================================================
# IMPORTS:
# ==================================================

import os
import json
import time
import random
import groq                     # type: ignore
from google import genai
from dotenv import load_dotenv

# ==================================================
# Client-Initialisierung:
# ==================================================

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client  = genai.Client(api_key=gemini_api_key)
GEMINI_MODEL = "gemini-3-flash-preview"

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=groq_api_key)
GROQ_MODEL = "llama-3.3-70b-versatile"

# ==================================================
# API-Calling-Error-Handling-Functions:
# ==================================================    

def call_gemini(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return response.text

def call_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

QUOTA_PHRASES = ["daily limit", "ratequotalimitreached"]

def is_quata_error(e: Exception) -> bool:
    return any(phrase in str(e).lower() for phrase in QUOTA_PHRASES)

def call_with_retry(func, max_retries=5) -> str:
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if is_quata_error(e):
                raise
            
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"[Retry {attempt+1}] {type(e).__name__}, waiting {wait:.2f}s...")
            time.sleep(wait)
    
    raise Exception("Max retries exceeded.")

PROVIDERS = [
    ("Gemini", call_gemini),
    ("Groq",   call_groq)
]

EXHAUSTED_PROVIDERS = set()

def call_with_fallback(prompt: str) -> str:
    for name, provider_func in PROVIDERS:
        if name in EXHAUSTED_PROVIDERS:
            print(f"[Provider] Skipping {name} (known exhausted)...")
            continue
        try:
            print(f"[Provider] Trying {name}...")
            return call_with_retry(func=lambda p=provider_func: p(prompt))
        except Exception as e:
            if is_quata_error(e):
                print(f"[Provider] {name} quata reched, blacklisting...")
                EXHAUSTED_PROVIDERS.add(name)
            else:
                print(f"[Provider] {name} exhausted ({type(e).__name__}), switching...")
            continue
    
    raise Exception("All providers exhausted.")

# ==================================================
# Data-Assembly:
# ==================================================

def build_report_data() -> dict:
    """
    Reads ranked_papers.json and summary.json
    and assembles the data dict used in the report prompt.
    Retrieve Information from trends.json and gap_analysis.json
    """
    with open("data/processed/ranked_papers.json", "r", encoding="utf-8") as f:
        ranked = json.load(f)
 
    with open("data/processed/summary.json", "r", encoding="utf-8") as f:
        summary_file = json.load(f)
    
    with open("data/processed/clinical_gap_analysis.json", "r", encoding="utf-8") as f: 
        gap_analysis_file = json.load(f)
    
    with open("data/processed/trends.json","r",encoding="utf-8") as f: 
        trends_file = json.load(f)
 
    intervention     = ranked["intervention"]
    papers           = ranked["papers"]
    summary          = summary_file.get("summary", {})
    confidence_score = ranked.get("confidence_score", "not calculated")
    confidence_label = ranked.get("confidence_label", "")
    gaps             = gap_analysis_file.get("clinical_gaps", [])
    trends           = trends_file.get("trends",[])
    # Format source list for the prompt
    sources = []
    for p in papers:
        authors = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors += " et al."
        sources.append(
            f"- {authors} ({p.get('pubdate', 'n.d.')}): {p.get('title', 'No title')}. "
            f"{p.get('journal', '')}. PMID: {p.get('pmid', '?')}"
        )
 
    return {
        "question":         f"What scientific evidence exists for the intervention '{intervention}' in the context of longevity?",
        "intervention":     intervention,
        "verified_facts":   summary.get("key_findings", []),
        "overall_evidence": summary.get("overall_evidence_level", "unknown"),
        "mechanisms":       summary.get("mechanisms", []),
        "limitations":      summary.get("limitations", []),
        "summary_text":     summary.get("summary", ""),
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "sources":          sources,
        "paper_count":      len(papers),
        "gaps":             gaps,
        "trends":           trends

    }
 
 
# ==================================================
# LLM-Report-Generation:
# ==================================================
 
def generate_report(data: dict) -> str:
    """
    Sends the prompt to Gemini and returns the Markdown report.
    """
    verified_facts_str = "\n".join(f"- {f}" for f in data["verified_facts"])
    mechanisms_str     = "\n".join(f"- {m}" for m in data["mechanisms"])
    limitations_str    = "\n".join(f"- {l}" for l in data["limitations"])
    sources_str        = "\n".join(data["sources"])
 
    prompt = f"""
You are a scientific author specializing in biogerontology and longevity research.
 
Task:
Write a scientific report in English based on the provided data.
 
Writing rules:
1. Use exclusively formal and objective language.
2. Use neutral but analytical expressions.
3. Write complete, grammatically correct sentences.
4. No bullet points in the body text.
5. No colloquial language.
6. No first-person perspective or personal opinions.
7. No unsubstantiated claims.
8. Clearly separate factual presentation from interpretation.
9. Do not fabricate sources — use only the provided reference list.
10. Be critical and differentiated — do not overstate the evidence.
 
---
 
Research question:
{data["question"]}
 
Overall evidence summary:
{data["summary_text"]}
 
Overall evidence level:
{data["overall_evidence"]}
 
Verified key findings:
{verified_facts_str}
 
Biological mechanisms:
{mechanisms_str}
 
Limitations of the evidence base:
{limitations_str}
 
Confidence Score: {data["confidence_score"]} ({data["confidence_label"]})
Number of papers evaluated: {data["paper_count"]}

Clinical Gap: 
{data["gaps"]}

Current Trends results: 
{data["trends"]}
 
Available sources:
{sources_str}
 
---
 
Use exactly the following structure:
 
# Scientific Report: {data["intervention"]}
 
## Abstract
Brief summary (3-5 sentences) of the methodology and key findings.
 
## Introduction
Contextualisation of the topic and relevance of the research question within longevity research.

## Trends
Create a timeline and contextualise the trends {data["trends"]} found. Make sure to name the sources as well.

## Methodology
Description of the data basis (PubMed search), filtering criteria, and evaluation logic.
 
## Results
Presentation of the verified, substantiated findings — structured by evidence level and study type.
 
## Biological Mechanisms
Description of the identified mechanisms of action of the intervention.

## Clinical Gap Analysis
Description of the identified clinical gaps {data["gaps"]}. Make sure to include the Citation of found gaps and the Suggestions to close the gaps

## Discussion
Interpretation of the results and critical examination of the limits of their validity.
 
## Confidence Score Assessment
Contextualisation of the score ({data["confidence_score"]} — {data["confidence_label"]}):
Briefly explain the calculation basis and interpret the score in the context of the available evidence.
 
## Conclusion
Answer to the research question in condensed form (3-5 sentences).
 
## References
List of sources used — exclusively from the provided references, none fabricated.
 
---
 
Formatting: The entire response must be in Markdown format.
"""
 
    response_text = call_with_fallback(prompt=prompt)
    
    return response_text
 
 
# ==================================================
# Save Functions
# ==================================================
 
def save_markdown(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[report_agent] Markdown saved: {path}")
 
 
def save_pdf(markdown_text: str, path: str) -> None:
    """
    Converts Markdown to PDF via pypandoc.
    Requires a local LaTeX installation (e.g. TeX Live).
    Fails silently if LaTeX is not available.
    """
    try:
        import pypandoc
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pypandoc.convert_text(
            markdown_text,
            "pdf",
            format="md",
            outputfile=path,
            extra_args=["--standalone", "--toc"]
        )
        print(f"[report_agent] PDF saved: {path}")
    except ImportError:
        print("[report_agent] pypandoc not installed — skipping PDF export.")
    except Exception as e:
        print(f"[report_agent] PDF export failed: {e}")
        print("[report_agent] Note: PDF export requires a local LaTeX installation.")
 
 
# ==================================================
# Pipeline Function (called by main.py)
# ==================================================
 
def create_report(save_pdf_output: bool = False) -> str:
    """
    Full report workflow:
    1. Assemble data from JSON files
    2. Generate report via LLM
    3. Save as Markdown
    4. Optionally save as PDF
    Returns the Markdown text.
    """
    print("[report_agent] Generating scientific report...")
 
    data          = build_report_data()
    markdown_text = generate_report(data)
 
    save_markdown(markdown_text, path="outputs/report.md")
 
    if save_pdf_output:
        save_pdf(markdown_text, path="outputs/report.pdf")
 
    print("[report_agent] Report created successfully!")
    return markdown_text
 
 
if __name__ == "__main__":
    create_report(save_pdf_output=True)