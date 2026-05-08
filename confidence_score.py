# ==================================================
# confidence_score.py
# Calculates a normalized confidence score (0–1) based on the ranked papers for an intervention.
# Input:  data/processed/ranked_papers.json
# Output: data/processed/ranked_papers.json (score is added)
# ==================================================
# IMPORTS:
# ==================================================

import json
import math

# ==================================================
# Weighting-Tables:
# ==================================================

STUDY_TYPE_WEIGHTS: dict[str, float] = {
    "Systematic reviews & meta-analyses":       1.0,
    "Randomised controlled trials (RCTs)":      5/6,
    "Observational / epidemiological studies":  4/6,
    "Animal model studies (in vivo)":           3/6,
    "Cell culture / in vitro studies":          2/6,
    "In silico / computational predictions":    1/6,
    "Hypothesis / perspective / commentary":    0.0,
}

EFFECT_TYPE_WEIGHTS: dict[str, float] = {
    "lifespan":      1.0,
    "healthspan":    5/6,
    "functional":    4/6,
    "biomarker":     3/6,
    "mechanistic":   2/6,
    "computational": 1/6,
    "unclear":       0.0,
}

RESULT_WEIGHTS: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral":  -0.5,
    "unclear":   0.05,
}

# ==================================================
# Score-Calculation:
# ==================================================

def compute_confidence_score(papers: list[dict]) -> float:
    """
    Calculates the confidence score across all papers.

    Formula per paper:
        contribution = study_weight * result_weight * effect_weight

    Normalization:
        raw_sum = sum(contributions) / sqrt(total_study_weight)
        score   = sigmoid(raw_sum)  →  range(0, 1)

    The sqrt normalization dampens the effect of many papers with
    a low level of evidence without completely eliminating it.
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for paper in papers:
        w = STUDY_TYPE_WEIGHTS.get(paper.get("study_type", ""), None)
        r = RESULT_WEIGHTS.get(paper.get("study_result", ""), None)
        e = EFFECT_TYPE_WEIGHTS.get(paper.get("effect_type", ""), None)

        # Skip the paper if a key is missing or cannot be mapped
        if w is None or r is None or e is None:
            print(f"[SKIP] Unknown key in PMID {paper.get('pmid', '?')} "
                  f"— study_type='{paper.get('study_type')}' "
                  f"study_result='{paper.get('study_result')}' "
                  f"effect_type='{paper.get('effect_type')}'")
            continue

        weighted_sum  += w * r * e
        total_weight  += w

    if total_weight == 0.0:
        print("[WARNING] No paper could be evaluated — the score is set to 0.5.")
        return 0.5

    # Normalisierung + Sigmoid
    normalized = weighted_sum / math.sqrt(total_weight)
    score = 1.0 / (1.0 + math.exp(-normalized))
    return round(score, 4)


def interpret_score(score: float) -> str:
    """Returns a human-readable interpretation of the score."""
    if score >= 0.75:
        return "high — strong, consistent evidence"
    elif score >= 0.60:
        return "moderate — several positive findings, but with limitations"
    elif score >= 0.45:
        return "low — conflicting or weak evidence"
    else:
        return "very low — little or no reliable evidence"


# ==================================================
# Pipeline function (called by main.py)
# ==================================================

def calculate_confidence_score() -> dict:
    """
    Loads ranked_papers.json, calculates the score,
    writes it back, and returns the result dictionary.
    """
    print("[confidence_score] Calculate confidence score...")

    with open("data/processed/ranked_papers.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    papers       = dataset["papers"]
    intervention = dataset["intervention"]

    score       = compute_confidence_score(papers)
    label       = interpret_score(score)
    paper_count = len(papers)

    result = {
        "confidence_score":       score,
        "confidence_label":       label,
        "papers_used":            paper_count,
    }

    # Write the score to the dataset and save it
    dataset["confidence_score"] = score
    dataset["confidence_label"] = label

    with open("data/processed/ranked_papers.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"[confidence_score] Score: {score}  ({label})")
    print(f"[confidence_score] Based on {paper_count} Paper(s) for '{intervention}'")

    return result


if __name__ == "__main__":
    calculate_confidence_score()