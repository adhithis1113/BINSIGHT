
import os
from auth import load_users, verify_password

email = "official.demo@binsight.test"
password = "OfficialDemo123"

print("Checking users.json:", os.path.abspath("users.json"))

users = load_users()
user = users.get(email)

print("Account exists:", user is not None)

if user:
    print("Account role:", user.get("role"))
    print(
        "Password matches:",
        verify_password(password, user.get("password_hash", ""))
    )
