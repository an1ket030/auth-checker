import requests
import sys

BASE_URL = "http://localhost:8000/api/v1"

def test_auth():
    print("Testing Authentication...")
    
    # 1. Register
    username = f"user_{import_uuid()}"
    email = f"{username}@example.com"
    password = "securepass123"
    
    print(f"Registering {username}...")
    try:
        res = requests.post(f"{BASE_URL}/register", json={
            "username": username,
            "email": email,
            "password": password
        })
        if res.status_code != 200:
            print(f"Registration failed: {res.text}")
            return
        data = res.json()
        print("Registration success! Token received.")
        token = data["access_token"]
        
        # 2. Login
        print("Testing Login...")
        res = requests.post(f"{BASE_URL}/login", json={
            "username": username,
            "password": password
        })
        if res.status_code != 200:
            print(f"Login failed: {res.text}")
            return
        print("Login success!")
        
        # 3. Access Protected Route (History)
        print("Testing Protected Route (History)...")
        headers = {"Authorization": f"Bearer {token}"}
        res = requests.get(f"{BASE_URL}/history", headers=headers)
        if res.status_code != 200:
            print(f"History access failed: {res.text}")
            return
        print(f"History access success! Records: {len(res.json())}")
        
    except Exception as e:
        print(f"Test failed with exception: {e}")

def import_uuid():
    import uuid
    return str(uuid.uuid4())[:8]

if __name__ == "__main__":
    test_auth()
