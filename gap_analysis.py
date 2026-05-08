#libraries
import ollama
import json
#first draft code: 
def first_draft(data):
    formatted_data = json.dumps(data, indent=2) 
    first_draft_prompt = """
You are a Senior Biogerontologist. 
Your task is to write a CLINICAL GAP ANALYSIS based EXCLUSIVELY on the provided JSON data. 

MUST USED DATA FIELDS IN THE JSON DATA FOR REASONING:
1. pmid,title,journal,pubdate,authors: Use {"title":"...","pmid":"..."} for every citation
2.study_type/study_design: Classify what kind of evidence exits (RCT,animal,in-vitro,...)
3.effect_type ["lifespan"|"healthspan"|"functional"|"biomarker"|"mechanistic"|"computational"|"unclear"]: Make clear Distinctions between the effect_types
3. evidence_level ["low"|"very_low"|"moderate"|"high"|"high"] and evidence_rank [1-6]: Use these to determine how critical the gap_analysis should be
4. sample_size_estimate["small"|"large"|"high"|"unknown"] : Fkag underpowered studies explicitly
5. key_limitations (list) : PRE-DETERMINED weaknesses. Every limitation in the top-3 papers by relevance_score MUST appear as a seperate entry in clinical_gaps
6. Strengths (list): Aknowledge genuine methodological strengths DO NOT overstate them. 
7. Intervention_relation["direct"|"indirect"|"mention"]: 
    - "direct" = study directly tests the ageing intervention or close derivative. 
    - "indirect" = study tests a mechanistically related pathway (TOR,etc)
    - "mention" = intervention appears in passing only. 
    papers with relation = "mention" must NEVER be cited as GOOD
8. relevance_score [0.0-1.0]: Papers with a score < 0.3 are background context ONLY, not primary sources. The top 3 Papers by relevance_score carry the MOST evidentiary weight.
9. Justification: Use this to verify whether indirect papers are properly qualified. 

EVIDENCE FLAGS USE EXACTLY AS WRITTEN: 
- CRITICAL GAP: No human RCT data: animal/mechanistic evidence only. 
- TRANSLATION GAP: Positive animal results, that are however not replicated in humans. 
- CONFLICTING EVIDENCE: Studies reach opposite conclusions on the same endpoint. 
- LOW EVIDENCE: Fewer than 3 peer-reviewed direct studies. 

MUST FOLLOW FORMATTING RULES: 
1. Cite EVERY factual claim with {"title":"...","pmid":"..."}.
2. Papers with intervention_relation = "mention" must not be cited for efficacy. 
3. If no data exists for a field, write "DATA NOT FOUND".
4. Assign evidence flags based on actual study_type + evidence_level, NOT just study_result. 
5. the clinical gaps section must include one entry per distinct key_limitation found across top 3 papers by relevance score.


STRICT OUTPUT RULE: 
Return ONLY a valid JSON object. Do not include any conversational text, markdown headers, or explanations.

JSON SCHEMA:
{
  "meta": {
    "intervention": "string",
    "report_date": "YYYY-MM-DD",
    "total_papers_analysed": integer,
    "direct_evidence_papers": integer,
    "indirect_evidence_papers": integer,
    "mention_only_papers": integer,
    "average_relevance_score": float,
    "evidence_cieling": "string — highest study_type present (e.g. Animal model in vivo)"
  },
  "summary": {
    "overall_evidence_rank": "string",
    "evidence_level": "string — e.g. Level 4 (animal studies only)",
    "confidence_in_intervention": "HIGH | MODERATE | LOW | VERY LOW",
    "human_trial_data_found": true | false,
    "key_takeaways": ["string — max 3 items, one sentence each"]
  },
  "clinical_gaps": [
    {
      "gap_id": "CG-01",
      "gap_type": "CRITICAL GAP | TRANSLATION GAP | CONFLICTING EVIDENCE | LOW EVIDENCE | EMERGING SIGNAL",
      "domain": "string — e.g. Human Trial Data, Model Organism Generalisability, Safety",
      "description": "string — precise explanation referencing the specific source limitation",
      "source_limitation": "string — exact key_limitation text from the paper",
      "affected_population": "string",
      "citation": {"title": "string", "pmid": "string"},
      "to_close_gap": "string — minimum study design required"
    }
  ],
  "safety_profile": {
    "reported_adverse_events": ["string — include PMID for each"],
    "immunosuppression_risk": "DOCUMENTED | SUSPECTED | NOT ASSESSED",
    "long_term_safety_data": "AVAILABLE | LIMITED | ABSENT",
    "source_pmids": ["string"]
  },
  "conclusion": {
    "readiness_for_human_trials": "READY | CONDITIONAL | NOT READY",
    "rationale": "string — one paragraph justifying the verdict",
    "suggested_next_steps": "string"
  }
}
"""
    user_prompt = f"""
    Here are the provided ranked research papers on the intervention: "{data.get('intervention')}" 
    RESEARCH DATA: 
    {formatted_data}
    FINAL INSTRUCTION: 
    Generate the report now. REMEMBER TO INCLUDE THE DATA FIELDS AND FOLLOW THE STRICT RULES"""
    reponse = ollama.chat(
        model = "command-r7b",
        format = "json",
        messages = [
            {"role":"system","content":first_draft_prompt},
            {"role": "user","content":user_prompt}
        ],
        options={
            "temperature": 0.2,
            "num_ctx" : 16384
        }
    )
    report = reponse["message"]["content"]
    return report
#auditor: 
def auditor(draft,data): 
    formatted_data = json.dumps(data, indent=2)
    instructions = """You are a Clinical Audit Specialist.
    Your ONLY Job is to verfy the DRAFT against the SOURCE DATA. 
    
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
    
    OUTPUT JSON SCHEMA:
    {
    "audit_metadata":{
    "score":0-10,
    "status": "PASS | FAIL",
    "errirs": integer
    },
    "validated_facts:[
    {
    "claim":"string - correct claim from the draft",
    "pmid":"string"
    }
    ],
    "discrepancies:[
    {
    "claim": "string - exact mention claimed from the draft",
    "actual_fact": "string - correct information from the source data",
    "source_field":"String - which JSON field was violated",
    "Error":"RELATION VIOLATION|HALLUCINATION|CITATION ERROR|OPTIMISM BIAS| FLAG MISUSE| SCORE INFLATION",
    }
    ],
    "missing_gaps": [
    {
    "title":"string",
    "pmid":"string",
    "missing_key_limitation"string - exact text from key_limitations field",
    "gap_type:"string"
    }]
    }"""
    input = f"""
        ORIGINAL SOURCE DATA (GROUND TRUTH): 
        {formatted_data}
        DRAFT REPORT TO AUDIT: 
        {draft}
    """
    response = ollama.chat(
        model = "medgemma:4b",
        format = "json",
        messages=[
            {"role":"system","content":instructions},
            {"role":"user","content":input}
        ],
        options = {"temperature": 0.1,"num_ctx" : 16384,"keep_alive":"3m"}
    )
    return response["message"]["content"]
def merger(draft,data,audit): 
    formatted_data = json.dumps(data, indent=2)
    instructions = """You are a final medical Editor. Your task is to produce the definitive Clinical Gap Analysis JSON. 
    Inputs: SOURCE DATA (Ground Truth), DRAFT REPORT (Template), AUDIT (Corrections)
    TRUTH HIERARCHY: 
    1.SOURCE DATA: Ground truth, never contradicted
    2.AUDIT DATA: Every discrepancy must be corrected. 
    3. DRAFT REPORT: use its structure; replace flagged text with correct facts. 

    MERGING CRITERIA: 
    1. SCHEMA: Output must conform exactly to the Drafter's JSON schema. 
    2. RELATION VIOLATIONS: Remove any Infomration with citation to a "mention" paper. 
    3. HALLUCINATIONS: Add a new clinical_gap entry for each item in audit "missing_gaps".
    4. FLAG MISUSE: Replace with audit's suggested flag.
    5. CITATION: Every citation must exactly match source data (title,PMID)

    RETURN ONLY a valid JSON Object. 
"""
    input = f"""
            1. ORIGINAL SOURCE DATA (Ground Truth)
            {formatted_data}
            2. DRAFT REPORT (Template)
            {draft}
            3. AUDIT DATA (Improvements)
            {audit}
            FINAL TASK: Apply the Audit corrections to the Draft using the Source Data as the final authority. 
            OUTPUT: finalized JSON
    """
    response = ollama.chat(
        model = "medgemma:4b",
        format = "json",
        messages=[
            {"role":"system","content":instructions},
            {"role":"user","content":input}
        ],
        options = {"temperature": 0.1,"num_ctx" : 16384,"keep_alive":"3m"}
    )
    return response["message"]["content"]
def main(): 
    print("Accessing ranked_papers.json...")
    with open('data/processed/ranked_papers.json','r',encoding='utf-8') as file:
        data = json.load(file)
    print("Success")
    print("Creating Draft...")
    draft = first_draft(data)
    print("Sucess")
    print("Reviewing Draft...")
    audit_data = {
        "intervention":data.get("intervention"),
        "papers":[
            {
                "pmid":paper.get("pmid"),
                "title":paper.get("title"),
                "evidence_rank":paper.get("evidence_rank"),
                "study_type":paper.get("study_type"),
                "key_limitations":paper.get("key_limitations"),
                "relevance_score":paper.get("relevance_score"),
                "intervention_relation":paper.get("intervention_relation")
            }for paper in data.get("papers",[])
        ]
    }
    audit = auditor(draft,audit_data)
    print("Success\n")
    print("Creating final report...")
    final = merger(draft,audit_data,audit)
    print("Success")
    print("Saving Clinical_gap_analysis")
    final_data = json.loads(final)
    output_path = 'data/processed/clinical_gap_analysis.json'
    with open(output_path,'w',encoding='utf-8') as f: 
        json.dump(final_data,f,indent=4,ensure_ascii=False)
    print("Successfully saved analysis under data/processed")
if __name__ == "__main__": 
    main()