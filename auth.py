
import os
import json
import hmac
import secrets
import bcrypt
import streamlit as st

AUTH_FILE = "users.json"


def load_users():
    """Load registered accounts from the local JSON file."""
    if not os.path.exists(AUTH_FILE):
        return {}

    try:
        with open(AUTH_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def save_users(users):
    """Save account records to the local JSON file."""
    temp_file = AUTH_FILE + ".tmp"

    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)

    os.replace(temp_file, AUTH_FILE)


def hash_password(password):
    """Return a bcrypt password hash."""
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password, password_hash):
    """Verify a password against its stored hash."""
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )
    except (ValueError, TypeError):
        return False


def create_citizen_account(name, email, password):
    """Register a citizen account."""
    name = name.strip()
    email = email.strip().lower()

    if not name or not email or not password:
        return False, "Please fill in every field."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if len(password.encode("utf-8")) > 72:
        return False, "Password must be no longer than 72 UTF-8 bytes."

    users = load_users()

    if email in users:
        return False, "An account with this email already exists."

    users[email] = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "role": "citizen"
    }

    save_users(users)
    return True, "Account created successfully."


def authenticate(email, password):
    """Authenticate an existing citizen or official account."""
    email = email.strip().lower()
    user = load_users().get(email)

    if not user or not verify_password(
        password, user.get("password_hash", "")
    ):
        return None

    return {
        "name": user["name"],
        "email": user["email"],
        "role": user["role"]
    }


def create_official_account(email, name, password):
    """
    Provision an official account for an administrator.

    Do not expose this function through public signup.
    """
    email = email.strip().lower()
    name = name.strip()

    if not email or not name or not password:
        raise ValueError("Email, name and password are required.")

    if len(password) < 5:
        raise ValueError("Official passwords must contain at least 5 characters.")

    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password must be no longer than 72 UTF-8 bytes.")

    users = load_users()

    if email in users:
        raise ValueError("An account with this email already exists.")

    users[email] = {
        "name": name,
        "email": email,
        "password_hash": hash_password(password),
        "role": "official"
    }

    save_users(users)
