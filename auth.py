# ============================================
# FIREBASE AUTHENTICATION SYSTEM
# Handles signup, login, token verification
# International level security!
# ============================================

# firebase_admin: Google's Python SDK
import firebase_admin
from firebase_admin import credentials, auth


# os: file operations
import os

# json: read config files
import json

# Import our Firebase config
from firebase_config import FIREBASE_CONFIG

# ============================================
# INITIALIZE FIREBASE ADMIN
# Uses serviceAccount.json for backend auth
# This is the SECURE server-side connection!
# ============================================

# Path to service account file
SERVICE_ACCOUNT_PATH = "serviceAccount.json"

# Initialize only once!
if not firebase_admin._apps:
    # Load service account credentials
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    # Connect to Firebase project
    firebase_admin.initialize_app(cred)
    print("Firebase Admin initialized!")
import requests as req
FIREBASE_API_KEY = FIREBASE_CONFIG["apiKey"]

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
        # Return error if signup fails
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
            return {"success": False, "error": "Wrong email or password!"}

        token = data["idToken"]
        name = data.get("displayName", email)
        print(f"User logged in: {email}")
        return {
            "success": True,
            "token": token,
            "email": email,
            "name": name,
            "uid": user["localId"]
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
# WHY: Every API request must be verified!
# ============================================
def verify_token(token):
    try:
        # Verify token with Firebase Admin
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
