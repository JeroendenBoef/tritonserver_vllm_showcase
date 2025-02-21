import requests
import json

url = "http://localhost:8000/v2/models/llama3-8b-instruct/generate"

payload = {
    "text_input": "How do you fry an egg?",
    "parameters": {"stream": False, "temperature": 0, "max_tokens": 200},
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)

print(response.status_code)
print(response.json())
