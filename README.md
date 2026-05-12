# Longevity Intervention Researcher

A command-line tool that automatically retrieves and evaluates scientific literature for a given longevity intervention. It retrieves papers from PubMed, filters them for relevance, uses LLMs to classify, assess, and summarize the evidence, analyzes current trends on the topic, generates a confidence score, identifies gaps in the research and creates a clear PDF-report that presents the results in an illustrative manner.

---

The program runs through a fixed pipeline:
 
1. **Retrieve** - fetches papers from PubMed for the given intervention
2. **Validate** - filters out irrelevant papers using generated keywords
3. **Classify** - assigns study type, result, and effect type to each paper
4. **Assess** - evaluates methodological quality and relation to the intervention
5. **Score** - ranks papers by a weighted relevance score
6. **Summarize** - generates a structured evidence summary from the top papers
7. **Confidence Score** - creates a normalized confidence score (0.0 - 1.0)
8. **Clinical Gap Analysis** - generates a clinical gap analysis to fetched papers from PubMed
9. **Trends** - searches for the current trends of the given intervention (2024-2026)
10. **Create Report** - summarizes all results and generates a clear report

---

## Installation
```bash
git clone git@github.com:neelkrishnatray/MPI-Software-Internship.git
cd MPI-Software-Internship

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
```

Then open the `.env` file and enter your API keys.

---

## Usage
 
```bash
python main.py
```

You will be prompted to enter a longevity intervention:
```bash
[longevity_ai] Please enter the intervention you would like to know more about:
```

Results are then saved in ```outputs/```.

---

## Project-Structure after a run

```bash
.
├── README.md
├── __pycache__/
│   └── [...]
├── confidence_score.py
├── data/
│   ├── processed/
│   │   └── [...]
│   └── raw
│       └── [...]
├── data_handling.py
├── gap_analysis.py
├── main.py
├── outputs
│   ├── report.md
│   └── report.pdf
├── report_agent.py
├── requirements.txt
├── trend.py
└── venv/
    └── [...]
```

--- 

## API-Keys

This project requires three API keys. After running `cp .env.example .env` , open `.env` and fill in:

```
CEREBRAS_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

- **Cerebras** - [cerebras.ai/inference](https://www.cerebras.ai/inference)
- **Gemini** – [aistudio.google.com](https://aistudio.google.com)
- **Groq** – [console.groq.com](https://console.groq.com)

  ---

## Limitations
This agentic workflow has the ability to deliver an accurate scientific report of ageing intervention, nevertheless it does come with its own limitations. For instance, the workflow only retrieves and evaluates papers from the PubMed database, which could reduce its accuracy if studies with significant importance to the intervention are not published in Pubmed. Subsequently, to reduce hallucinations and to preserve tokens, the number of papers accessed from Pubmed has been limited to 5 papers in total. As a final point the program should not be run more than 3 times a day in order to avoid reaching the maximum requests per day error from the API providers under the free trial.

  

