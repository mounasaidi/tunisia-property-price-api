import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "type": "louer",
    "room_count": 3,
    "bathroom_count": 2,
    "size": 120
}

response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Raw response:", response.text)

try:
    print(response.json())
except:
    print("❌ Pas de JSON retourné")