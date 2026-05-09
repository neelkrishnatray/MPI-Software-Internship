# Longevity Intervention Researcher

A command-line tool that automatically retrieves and evaluates scientific literature for a given longevity intervention. It retrieves papers from PubMed, filters them for relevance, uses LLMs to classify, assess, and summarize the evidence, analyzes current trends on the topic, generates a confidence score, identifies gaps in the research and creates a clear PDF-report that presents the results in an illustrative manner.

---

The program runs through a fixed pipeline:
 
1. **Retrieve** – fetches papers from PubMed for the given intervention
2. **Validate** – filters out irrelevant papers using LLM-generated keywords
3. **Classify** – assigns study type, result, and effect type to each paper
4. **Assess** – evaluates methodological quality and relation to the intervention
5. **Score** – ranks papers by a weighted relevance score
6. **Summarize** – generates a structured evidence summary from the top papers
7. [...]

[...]

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
python [...]
```

You will be prompted to enter a longevity intervention:

[...]

Results are saved in [...]. Final summary can be found in [...].

---

## Project Structure

```bash
[tree]
```

--- 

## API Keys

This project requires three API keys. After running `cp .env.example .env` , open `.env` and fill in:

```
CEREBRAS_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here

```

- **Gemini** – [aistudio.google.com](https://aistudio.google.com)
- **Groq** – [console.groq.com](https://console.groq.com)
- **Cerebras** - [cerebras.ai/inference](https://www.cerebras.ai/inference)
