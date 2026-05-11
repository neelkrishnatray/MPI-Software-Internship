# ==================================================
# report_agent.py
# Generates the final research report
# and saves it as Markdown and PDF.
#
# Input:  data/processed/ranked_papers.json
#         data/processed/summary.json
# Output: outputs/report.md
#         outputs/report.pdf
# ==================================================
# IMPORTS:
# ==================================================

import os                       # type: ignore
import json                     # type: ignore
import time                     # type: ignore
import random                   # type: ignore
import groq                     # type: ignore
from google import genai        # type: ignore
from dotenv import load_dotenv  # type: ignore
from cerebras.cloud.sdk import Cerebras     # type: ignore

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

cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
cerebras_client = Cerebras(api_key=cerebras_api_key)
CEREBRAS_MODEL = "gpt-oss-120b"

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

def call_cerebras(prompt:str)->str: 
    response = cerebras_client.chat.completions.create(
        model = CEREBRAS_MODEL,
        messages = [{"role":"user","content":prompt}]
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
    ("Gemini",call_gemini),
    ("Groq",call_groq),
    ("Cerebras",call_cerebras)
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
                print(f"[Provider] {name} quota reached, blacklisting...")
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

STRICT LENGTH RULES — these are hard limits, not suggestions:
- Every section must be as concise as possible. Dense, information-rich prose only.
- No filler phrases, only few repetition of findings already stated in a prior section if necessary.
- Each section limit is stated below.
 
Use exactly the following structure:
 
# Scientific Report: {data["intervention"]}
 
## Abstract
LIMIT: 3 sentences. Cover: research question, data basis, key finding, and evidence level.
 
## Introduction
LIMIT: 3 sentences. State why this intervention is relevant to longevity research. Do not repeat the abstract.

## Trends
LIMIT: 1-2 sentences per trend. Present as a compact chronological overview. Name source context per trend.

## Methodology
LIMIT: 3 sentences. State: data source (PubMed), number of papers, filtering logic, and evaluation criteria.
 
## Results
LIMIT: 2 sentences per paper (max). Group by evidence level. State study type, result, and intervention relation.

## Biological Mechanisms
LIMIT: 4 sentences total. Name the mechanisms; do not elaborate beyond what the data supports.

## Clinical Gap Analysis
LIMIT: 1 focused paragraph per gap (3-4 sentences). Include: gap type, citation (title + PMID), and one concrete suggestion to close it.

## Discussion
LIMIT: 4 sentences. Interpret results critically. Do not repeat findings from Results — add interpretive value only.
 
## Confidence Score Assessment
LIMIT: 3 sentences. State the score, explain the calculation basis briefly, interpret in context of evidence.
 
## Conclusion
LIMIT: 3 sentences. Directly answer the research question. Do not introduce new content.
 
## References
One line per source. Use exclusively the provided references — none fabricated..
 
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
    Converts Markdown to HTML to PDF via weasyprint.
    Fails silently if weasyprint or markdown not installed.
    """
    try:
        import markdown as md       # type: ignore
        from weasyprint import HTML # type: ignore
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Markdown -> HTML -> PDF
        html_body = md.markdown(markdown_text, extensions=["tables", "fenced_code"])
        html_full = f"""
        <html><head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 18mm 20mm 18mm 20mm;
            }}

            body {{
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 8.5pt;
                line-height: 1.45;
                color: #1a1a1a;
                max-width: 100%;
            }}

            h1 {{
                font-size: 13pt;
                font-weight: 700;
                color: #1a2e4a;
                margin: 0 0 6px 0;
                padding-bottom: 5px;
                border-bottom: 2px solid #1a2e4a;
                letter-spacing: 0.3px;
            }}

            h2 {{
                font-size: 9pt;
                font-weight: 700;
                color: #1a2e4a;
                margin: 10px 0 3px 0;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                border-left: 3px solid #2980b9;
                padding-left: 6px;
            }}

            h3 {{
                font-size: 8.5pt;
                font-weight: 600;
                color: #2c3e50;
                margin: 6px 0 2px 0;
            }}

            p {{
                margin: 0 0 4px 0;
                text-align: justify;
            }}

            ul, ol {{
                margin: 2px 0 4px 0;
                padding-left: 16px;
            }}

            li {{
                margin-bottom: 1px;
            }}

            code {{
                background: #f0f0f0;
                padding: 1px 4px;
                border-radius: 2px;
                font-size: 7.5pt;
            }}

            pre {{
                background: #f0f0f0;
                padding: 6px 8px;
                border-radius: 3px;
                font-size: 7pt;
            }}
        </style>
        </head>
        <body>{html_body}</body>
        </html>
        """
        HTML(string=html_full).write_pdf(path)
        print(f"[report_agent] PDF saved: {path}")
    except ImportError:
        print("[report_agent] weasyprint / markdown not installed — skipping PDF export.")
    except Exception as e:
        print(f"[report_agent] PDF export failed: {e}")
 
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