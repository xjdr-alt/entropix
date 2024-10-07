import requests
import json
from entropix.prompts import prompt

url = 'http://127.1.1.1:8000/generate'
headers = {'Content-Type': 'application/json'}
data = {"stream": True, "prompt": prompt}
res = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

if res.status_code == 200:
  for chunk in res.iter_content(chunk_size=4):
    if chunk:
      print(chunk.decode('utf-8'), end='', flush=True)
else:
  print(f"Error: {res.status_code}")
  print(res.text)
