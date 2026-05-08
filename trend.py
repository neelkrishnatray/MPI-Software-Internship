import os
import ollama
import json
from ollama import Client
from dotenv import load_dotenv

#Retrieving API_KEY
load_dotenv()
api_key = os.getenv("OllAMA_API_KEY")
#Accessing the websearch function from Ollama
def websearch_results(query): 
    client = Client(
        headers={'Authorization': f'Bearer {api_key}'}
    )
    search_results = client.web_search(f"{query}research timeline milestones and announcement dates")
    return search_results
#transforming found data into a json format: 
def get_trends(search_results): 
    instructions = """You are a specialized Biotech Data Extractor. 
Your ONLY job is to convert search data into a specific JSON format.
DO NOT use your default summary format.
DO NOT include 'text' or 'key_points' keys.
Use ONLY the schema provided below."""
    prompt = f"""
    Based on these search results, extract the 5 most significant trends.
        
        SEARCH DATA:
        {search_results}

        OUTPUT MUST BE VALID JSON IN THIS EXACT FORMAT:
        {{
            "topic": "string",
            "trends": [
                {{
                    "trend_name": "string",
                    "summary": "2-sentence summary",
                    "earliest_mention_date": "MM-YYYY",
                    "trend_score": 1-10,
                    "source_context": "Short abstract"
                }}
            ]
        }}
    """
    response = ollama.chat(
        model="command-r7b",
        format="json",
        messages=[
            {"role":"system","content":instructions}, 
            {"role":"user","content":prompt}
        ],
        options = {"temperature":0.1}
    )
    return response["message"]["content"]
def main(query): 
    print(f"Searching Trends for {query}")
    search_results = websearch_results(query)
    print("Success")
    print("Summarzing Trends...")
    trends = get_trends(search_results)
    print("Success")
    print("Saving results...")
    trends_save = json.loads(trends)
    output_path = 'data/processed/trends.json'
    with open(output_path,'w',encoding='utf-8') as f: 
        json.dump(trends_save,f,indent=4,ensure_ascii=False)
    print("Success")
if __name__ == "__main__": 
    main("rapamycin_longetivity")