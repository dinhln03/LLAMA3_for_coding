import requests
import os
from dotenv import load_dotenv

def generate_instruction(api_key, code_snippet):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-3.5-turbo-instruct",
        "messages": [{
            "role": "user",
            "content": f"Summarize this code: {code_snippet}"
        }]
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception("API Error: " + response.status_code + " " + response.text)

def summarize_code_files(api_key, source_dir):
    files = os.listdir(source_dir)
    for file in files:
        if file.endswith(".txt") and 'content_' in file:
            with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as f:
                code_snippet = f.read()

            try:
                summary = generate_instruction(api_key, code_snippet)
                summary_file_path = os.path.join(source_dir, file.replace("content", "summary"))
                with open(summary_file_path, 'w', encoding='utf-8') as sf:
                    sf.write(summary)
                print(f"Generated summary for {file} and saved to {summary_file_path}")
            except Exception as e:
                print(f"Failed to generate summary for {file}: {str(e)}")

# Example usage
load_dotenv()
api_key = os.getenv('API_KEY')

source_dir = '/Users/vuh/Documents/chatgpt_code_summary/data/bigcode_the-stack-smol/content_test'
summarize_code_files(api_key, source_dir)
