# ==================================================
# main.py
# Pipeline Orchestrator for longevity_ai
#
# Workflow:
#   1. data_retrieval       - PubMed search + XML parsing
#   2. validate_data        - LLM keyword filter
#   3. classify_papers      - LLM classification per paper
#   4. assess_qualities     - LLM quality assessment + relation
#   5. score_papers         - Rule-based scoring + ranking
#   6. summarize_evidence   - LLM summary of top papers
#   7. confidence_score     - Normalized evidence score (0-1)
#   8. gap_analysis         - LLM analysis of clinial gaps
#   9. trend_search         - LLM search of trends on topic
#  10. create_report        — LLM-generated Markdown/PDF report
# ==================================================

import data_handling    as pipeline
import confidence_score as scoring
import report_agent     as reporter
import gap_analysis     as analysis
import trend            as trend_search

# ==================================================
# Pipeline:
# ==================================================

def run(intervention: str, save_pdf: bool = False) -> None:

    print(f"\n{'='*55}")
    print(f"  longevity_ai — Intervention: '{intervention}'")
    print(f"{'='*55}\n")

    # --- Schritt 1: Daten abrufen ---
    pipeline.make_dir_structure()
    pipeline.data_retrieval(intervention)

    # --- Schritt 2: Keyword-Validierung ---
    pipeline.validate_data()

    # --- Schritt 3: Klassifikation ---
    pipeline.classify_papers()

    # --- Schritt 4: Qualitätsbewertung & Relation ---
    pipeline.assess_qualities()

    # --- Schritt 5: Scoring & Ranking ---
    pipeline.score_papers()

    # --- Schritt 6: Evidenz-Zusammenfassung ---
    pipeline.summarize_intervention_evidence()

    # --- Schritt 7: Confidence Score ---
    scoring.calculate_confidence_score()

    # --- Schritt 8: Clinical Gap Analysis ---
    analysis.main()
 
    # --- Schritt 9: Trends suchen ---
    trend_search.main(intervention)
    
    # --- Schritt 10: Report ---
    reporter.create_report(save_pdf_output=save_pdf)

    print(f"\n{'='*55}")
    print(f"  Done. Output: outputs/report.md")
    if save_pdf:
        print(f"                outputs/report.pdf")
    print(f"{'='*55}\n")


# ==================================================
# ENTRY-POINT:
# ==================================================

if __name__ == "__main__":
    intervention = input("[longevity_ai] Please enter the intervention you would like to know more about: ").strip()

    if not intervention:
        print("[ERROR] No intervention specified.")
    else:
        run(intervention=intervention, save_pdf=True)