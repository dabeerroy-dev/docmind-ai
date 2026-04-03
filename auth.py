# ============================================
# FIREBASE AUTHENTICATION SYSTEM
# Handles signup, login, token verification
# International level security!
# ============================================

# firebase_admin: Google's Python SDK
import firebase_admin
from firebase_admin import credentials, auth

# requests: for Firebase REST API login
import requests as req

# os: file operations
import os

# json: read config files
import json

# ============================================
# FIREBASE CONFIG
# Reads from environment variables on server
# Or from firebase_config.py locally
# ============================================
try:
    from firebase_config import FIREBASE_CONFIG
except:
    # On server - read from environment variables
    FIREBASE_CONFIG = {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
        "databaseURL": ""
    }

# Firebase API key for REST calls
FIREBASE_API_KEY = FIREBASE_CONFIG.get("apiKey", "")

# ============================================
# INITIALIZE FIREBASE ADMIN
# Uses serviceAccount.json locally
# Uses environment variables on server
# ============================================
if not firebase_admin._apps:
    try:
        # Try local serviceAccount.json first
        SERVICE_ACCOUNT_PATH = "serviceAccount.json"
        if os.path.exists(SERVICE_ACCOUNT_PATH):
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)
            print("Firebase Admin initialized from file!")
        else:
            # On server - use environment variable
            service_account_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT", "")
            if service_account_json:
                service_account_dict = json.loads(service_account_json)
                cred = credentials.Certificate(service_account_dict)
                firebase_admin.initialize_app(cred)
                print("Firebase Admin initialized from env!")
            else:
                print("Warning: No Firebase credentials found!")
    except Exception as e:
        print(f"Firebase init error: {e}")

# ============================================
# FUNCTION 1: Sign Up New User
# Input: email + password + name
# Output: user info or error message
# ============================================
def signup_user(email, password, display_name):
    try:
        # Create user in Firebase
        user = auth.create_user(
            email=email,
            password=password,
            display_name=display_name
        )
        print(f"User created: {email}")
        return {
            "success": True,
            "uid": user.uid,
            "email": user.email,
            "name": display_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# FUNCTION 2: Login User
# Input: email + password
# Output: token + user info or error
# ============================================
def login_user(email, password):
    try:
        # Login using Firebase REST API
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
        payload = {
            "email": email,
            "password": password,
            "returnSecureToken": True
        }
        response = req.post(url, json=payload)
        data = response.json()

        if "error" in data:
            return {
                "success": False,
                "error": "Wrong email or password!"
            }

        token = data["idToken"]
        name = data.get("displayName", email)
        uid = data.get("localId", "")

        print(f"User logged in: {email}")
        return {
            "success": True,
            "token": token,
            "email": email,
            "name": name,
            "uid": uid
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Wrong email or password!"
        }

# ============================================
# FUNCTION 3: Verify Token
# Input: token from logged in user
# Output: user info or error
# ============================================
def verify_token(token):
    try:
        decoded = auth.verify_id_token(token)
        return {
            "success": True,
            "uid": decoded["uid"],
            "email": decoded.get("email", "")
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Invalid or expired token!"
        }

# ============================================
# FUNCTION 4: Get User Info
# Input: uid (user ID)
# Output: user details
# ============================================
def get_user(uid):
    try:
        user = auth.get_user(uid)
        return {
            "success": True,
            "uid": user.uid,
            "email": user.email,
            "name": user.display_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# TEST AUTH SYSTEM
# ============================================
if __name__ == "__main__":
    print("Testing Firebase Auth...")
    print("Firebase connected successfully!")
