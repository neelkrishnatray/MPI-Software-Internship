import os
from ollama import Client

# 1. Get the key from your environment
# Note: Ensure you ran 'export OLLAMA_API_KEY=your_key_here' in this terminal session
api_key = os.getenv("OLLAMA_API_KEY")

if not api_key:
    # Manual fallback for debugging—paste your key here if getenv fails
    api_key = "PASTE_YOUR_KEY_HERE_IF_GETENV_IS_EMPTY"

client = Client(
    headers={'Authorization': f'Bearer {api_key}'}
)

try:
    print("Checking internet access...")
    search_results = client.web_search(query='Current Tech Trends in 2026')
    
    print("Search successful! Analyzing results with Command R...")
    response = client.chat(
        model='command-r7b',
        messages=[
            {
                'role': 'system',
                'content': f'You are an expert analyst. Use these search results to answer: {search_results}'
            },
            {
                'role': 'user',
                'content': 'What are the top 3 trends found in these results?'
            }
        ]
    )
    
    print("\n--- 2026 TREND REPORT ---")
    print(response['message']['content'])

except Exception as e:
    print(f"\nOops! Something went wrong: {e}")