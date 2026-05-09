#IMPORTS:
#--------------------------------------------------------
import groq
import os
import random
from google import genai
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
import json
import time
#--------------------------------------------------------
#Client-Initialisierung:
#--------------------------------------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=gemini_api_key)
GEMINI_MODEL = "gemini-3-flash-preview"

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = groq.Groq(api_key=groq_api_key)
GROQ_MODEL = "llama-3.3-70b-versatile"

cerebras_api_key = os.getenv("CEREBRAS_API_KEY")
cerebras_client = Cerebras(api_key=cerebras_api_key)
CEREBRAS_MODEL = "gpt-oss-120b"
#-------------------------------------------------------
# API-Calling-Error-Handling-Functions: 
#-------------------------------------------------------
def call_gemini(prompt:str) -> str: 
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents = prompt
    )
    return response.text

def call_groq(prompt:str) -> str: 
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages = [{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content

def call_cerebras(prompt:str)->str: 
    response = cerebras_client.chat.completions.create(
        model = CEREBRAS_MODEL,
        messages = [{"role":"user","content":prompt}]
    )
    return response.choices[0].message.content
QUOTA_PHRASES = ["daily limit","ratequotalimitreached"]

def is_quota_error(e:Exception)->bool:
    return any(phrase in str(e).lower() for phrase in QUOTA_PHRASES)

def call_with_retry(func,max_retries=5) -> str: 
    for attempt in range(max_retries): 
        try: 
            return func() 
        except Exception as e: 
            if is_quota_error(e): 
                raise
            wait = (2**attempt)+random.uniform(0,1)
            print(f"[Retry {attempt+1}]{type(e).__name__},waiting {wait:.2f}s...")
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
            if is_quota_error(e):
                print(f"[Provider] {name} quata reched, blacklisting...")
                EXHAUSTED_PROVIDERS.add(name)
            else:
                print(f"[Provider] {name} exhausted ({type(e).__name__}), switching...")
            continue
    
    raise Exception("All providers exhausted.")

# #first draft code: 
def first_draft(data):
    formatted_data = json.dumps(data, indent=2) 
    prompt = f"""You are a Senior Biogerontologist. Your Task is to write a clinical gap analysis based exclusively on the data provided. 

SOURCE DATA: 
'''
{formatted_data}
'''

Writing rules: 
1. Make Sure to use all data Fields mentioned inside the SOURCE DATA. 
2. Every factual claim Must be cited with the title and pmid
3. Papers with intervention_relation = “mention” MUST not be cited for efficacy. 
4. If no data exists for a field write “DATA NOT FOUND”. 
5. Assign evidence flags based on actual study_type and evidence_level NOT only study_result. 
6. The Clinical gaps section must include one entry per distinct key_limitation found across the top 3 papers by relevance_score. 

Evidence Flags use exactly as written: 
- CRITICAL GAP: No human RCT data: animal/mechanistic evidence only. 
- TRANSLATION GAP: Positive animal results that are NOT mentioned to be replicated in humans.
- CONFLICTING EVIDENCE: studies reach opposite conclusions on the same endpoint. 
- LOW EVIDENCE: Fewer than 3 peer-reviewed studies.

STRICT OUTPUT RULE: 
Return ONLY a valid JSON object. Do not include any conversational text or markdown headers. 

JSON SCHEMA: 

{{
“Metadata”:{{
	“intervention”:”String”,
	“total_papers_analysed”:integer,
	“direct_evidence_papers”:integer,
	“indirect_evidence_papers”:integer,
	“mention_only_papers”:integer,
	“average_relevance_score”:float
	“main_study_type: “String-highest study_type present (e.g. Animal model in vitro) 
}}, 
“Summary”: {{
“overall_evidence_rank”:”string”,
“evidence_level”:”String - e.g. Level 4 (animal studies only)”,
“confidence_in_intervention”: “HIGH | MODERATE | LOW | VERY LOW”,
“human_trial_data_found”: true | false,
“key_takeaways”: [“String - max 3 items, one sentence each”]
}}, 
“clinical_gaps”:[
{{
“gap_id”: integer,
“gap_type”: “CRITICAL GAP | TRANSLATION GAP | CONFLICTING EVIDENCE | LOW EVIDENCE” , 
“domain”: “string -e.g. Human Trial Data, Safety, Model Organism”.
“description”: “string - precise explanation describing the specific source limitation”,
“Citation”:{{“title”:”string”,”pmid”:”string”}},
“to_close_gap”:”String- suggested minimum study design required to solve the gap”
}}]
}}

"""
    report = call_with_fallback(prompt=prompt)
    return report
#auditor: 
def auditor(draft,data): 
    formatted_data = json.dumps(data, indent=2)
    instructions = f"""You are a Clinical Audit Specialist.
    Your ONLY Job is to verfy the DRAFT against the SOURCE DATA. 
    SOURCE DATA: 
    '''
    {formatted_data}
    '''
    DRAFT REPORT TO AUDIT:
    '''
    {draft}
    '''

    STRICT RULES: 
    1. DO NOT WRITE A NEW REPORT.
    2. DO NOT PROVIDE A SUMMARY.
    3. RETURN ONLY A JSON OBJECT.
    
    WHAT TO CHECK: 
    - RELATION VIOLATION: A paper with intervention_relation = "mention" is cited as efficacy evidence. 
    - HALLUCINATION: Claim not supported by any paper field in the source
    - CITATION ERROR: Wrong PMID or TITLE citing a finding. 
    - OPTIMISM BIAS: positive mentioned of the source papers with evidence_level "low" or "very_low", or study_result = "unclear". 
    - FLAG MISUSE: Wrong evidence flag was applied. 
    - SCORE INFLATION: Papers with a relevance_score under 0.3 are cited as primary evidences. 
    
    Evidence Flags USE ONLY IF MISSED IN THE DRAFT IN missing_gaps : 
    - CRITICAL GAP: No human RCT data: animal/mechanistic evidence only. 
    - TRANSLATION GAP: Positive animal results that are NOT mentioned to be replicated in humans.
    - CONFLICTING EVIDENCE: studies reach opposite conclusions on the same endpoint. 
    - LOW EVIDENCE: Fewer than 3 peer-reviewed studies.

    OUTPUT JSON SCHEMA:
    {{
    "audit_metadata":{{
    "score":0-10,
    "status": "PASS | FAIL",
    "errors": integer
    }},
    "validated_facts:[
    {{
    "claim":"string - correct claim from the draft",
    "pmid":"string"
    }}
    ],
    "discrepancies:[
    {{
    "claim": "string - exact mention claimed from the draft",
    "actual_fact": "string - correct information from the source data",
    "source_field":"String - which JSON  data field was violated",
    "Error":"RELATION VIOLATION|HALLUCINATION|CITATION ERROR|OPTIMISM BIAS| FLAG MISUSE| SCORE INFLATION",
    }}
    ],
    "missing_gaps": [
    {{
    "title":"string",
    "pmid":"string",
    "missing_key_limitation"string - exact text from key_limitations field",
    "gap_type:"string"
    }}]
    }}"""
    report = call_with_fallback(prompt=instructions)
    return report

def merger(draft,audit): 
    instructions = f"""You are a final medical Editor. Your task is to produce the definitive Clinical Gap Analysis JSON. 
    Inputs: 
    DRAFT: 
    '''
    {draft}
    '''
    AUDIT DATA:
    '''
    {audit}
    '''
    TRUTH HIERARCHY: 
    1. AUDIT DATA: Every discrepancy must be corrected. 
    2. DRAFT REPORT: use its structure; replace flagged text with correct facts. 
    3. DO NOT MAKE ANY NEW FACTS USE ONLY THE PROVIDED SOURCES

    MERGING CRITERIA: 
    1. SCHEMA: Output must conform exactly to the Drafter's JSON schema. 
    2. RELATION VIOLATIONS: Remove any Infomration with citation to a "mention" paper. 
    3. HALLUCINATIONS: Add a new clinical_gap entry for each item in audit "missing_gaps".
    4. FLAG MISUSE: Replace with audit's suggested flag.
    5. CITATION: Every citation must exactly match source data (title,PMID)

    RETURN ONLY a valid JSON Object. 
"""
    report = call_with_fallback(prompt=instructions)
    return report
def extract_json(text): 
    try: 
        start_index = text.index("{")
        end_index = text.rindex("}")+1
        return text[start_index:end_index]
    except ValueError:
        return None
def save_file(text): 
    save_file = json.loads(text)
    output_path = 'data/processed/clinical_gap_analysis.json'
    with open(output_path,'w',encoding = 'utf-8') as f:
        json.dump(save_file,f,indent=4,ensure_ascii=False)
def main(): 
    print("[Clinical Gap Analysis] Accessing ranked_papers.json...")
    with open('data/processed/ranked_papers.json','r',encoding='utf-8') as file:
        data = json.load(file)
    print("[Clinical Gap Analysis] Creating Draft...")
    draft = first_draft(data)
    print("[Clinical Gap Analysis] Reviewing Draft...")
    audit = auditor(draft,data)
    print("[Clinical Gap Analysis] Creating final report...")
    final = merger(draft,audit)
    json_file = extract_json(final)
    save_file(json_file)
    print("[Clinical Gap Analysis] Creating Clinical Gap Analysis and saving file sucessfull!")

if __name__ == "__main__": 
    main()