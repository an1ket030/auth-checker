import requests
import sys

# Use the IP that the phone uses
BASE_URL = "http://11.12.4.147:8000/api/v1"

def test_login():
    print(f"Testing Login on {BASE_URL}...")
    
    # 1. Try to Register (in case user doesn't exist)
    reg_data = {
        "username": "test_verifier",
        "email": "verifier@example.com",
        "password": "Password123"
    }
    print(f"Attempting Register: {reg_data['username']}")
    try:
        r = requests.post(f"{BASE_URL}/register", json=reg_data, timeout=5)
        if r.status_code == 200:
            print("  [SUCCESS] Registration Successful")
        elif r.status_code == 400 and "already registered" in r.text:
             print("  [INFO] User already exists (Expected)")
        else:
             print(f"  [ERROR] Registration Failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"  [CRITICAL] Network Error during Register: {e}")
        return

    # 2. Try to Login
    login_data = {
        "username": "test_verifier",
        "password": "Password123"
    }
    print(f"Attempting Login: {login_data['username']}")
    try:
        r = requests.post(f"{BASE_URL}/login", json=login_data, timeout=5)
        if r.status_code == 200:
            token = r.json().get("access_token")
            print(f"  [SUCCESS] Login Successful! Token: {token[:10]}...")
        else:
            print(f"  [ERROR] Login Failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"  [CRITICAL] Network Error during Login: {e}")

if __name__ == "__main__":
    test_login()
