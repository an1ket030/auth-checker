# AuthChecker - Project Manual 🛡️

Congratulations! Your AI-Powered Product Authentication System is fully operational.
This document is your "Standard Operating Procedure" to run the project in the future.

## 🚀 Quick Start Guide

### 1. Start the Backend Server
This powers the AI Engine, Database, and API.
1.  Open Terminal in `d:\Projects\auth-checker`
2.  Run:
    ```bash
    .\run_server.bat
    ```
    *(Keep this window open!)*

### 2. Connect the Mobile App
Your Android App needs to talk to your laptop over WiFi.
1.  **WiFi Check:** Ensure Phone and Laptop are on the **Same WiFi**.
2.  **IP Check:** Open `mobile/config.js` and verify the IP address matches your laptop's current IPv4 address (run `ipconfig` in terminal to check).
    ```javascript
    export const API_URL = 'http://192.168.1.5:8000/api/v1'; // Update IP if it changes!
    ```

### 3. Run the App
**Option A: Installed APK (Best for Demo)**
1.  Just open the "AuthChecker" app on your phone.
2.  Login with `test_verifier` / `Password123` (or your new account).
3.  Scan away!

**Option B: Development Mode (To make changes)**
1.  Open Terminal in `d:\Projects\auth-checker\mobile`
2.  Run:
    ```bash
    npx expo start
    ```
3.  Scan the QR code with your phone.

---

## 🌟 Key Features Built
-   **AI OCR Engine:** Extracts text using PaddleOCR.
-   **Trust Score Algorithm:** Analyzes Serial Number format + Database Match + Scan Frequency to detect clones.
-   **Cloning Detection:** Flags items scanned >10 times as "Suspicious".
-   **Secure Auth:** JWT Tokens + BCrypt Hashing for industry-standard security.
-   **Premium UI:** Glassmorphism, Animations, and Haptic Feedback.

---

## ⚠️ Troubleshooting
-   **"Network Error":**
    -   Is the Server running?
    -   Are you on the same WiFi?
    -   Did your Laptop IP change? (Update `config.js` if it did).
    -   Is Windows Firewall blocking Python? (Allow it via "Allow an app through Windows Firewall").

**Good luck with your submission/demo!** 🚀
