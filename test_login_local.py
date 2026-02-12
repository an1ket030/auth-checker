import requests

URL = "http://127.0.0.1:8000/api/v1/login"
CREDENTIALS = {
    "username": "demo",
    "password": "password123"
}

try:
    print(f"Attempting login to {URL} with {CREDENTIALS}...")
    response = requests.post(URL, json=CREDENTIALS)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS: Login working locally!")
    else:
        print("\n❌ FAILURE: Login failed locally.")
        
except Exception as e:
    print(f"\n❌ ERROR: Could not connect to server. {e}")
