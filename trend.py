import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
 
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
def get_trends_with_search(query): 
    prompt = f"""You are Biotech specialist. Find the most significant/research trends for {query} in regards to ageing longevity (2024-2026). 
    You Must extract the 'earliest_mention_date' (MM-YYYY) for each trend. 
    Return ONLY a JSON object following this SCHEMA: 
    {{
    "topic":"string",
    "trends":[
    {{
    "trend_name":string,
    "summary":"String only 3 sentences"
    "earliest_mention_date": "MM-YYYY",
    "trend_score":1-10,
    "source_context": string-title of found Source
    }}
    ]
    }}
"""
    response = client.models.generate_content( 
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )
    )
    return response.text
def extract_json(text): 
    try: 
        start_index = text.index("{")
        end_index = text.rindex("}")+1
        return text[start_index:end_index]
    except ValueError: 
        return None
def main(query): 
    print(f"[Trend] Searching for trends: {query}" )
    text = get_trends_with_search(query)
    json_text = extract_json(text)
    save_file = json.loads(json_text)
    output_path = 'data/processed/trends.json'
    with open(output_path,'w',encoding='utf-8') as f: 
        json.dump(save_file,f,indent=4,ensure_ascii=False)
    print(f"[Trend] Finding and Saving Trends succesfull!")
