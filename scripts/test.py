import requests

query_url = "https://xeno-canto.org/api/v3/recordings?query=anura"
response = requests.get(query_url)

print(f"Status: {response.status_code}")
print(f"JSON: {response.json()}")
