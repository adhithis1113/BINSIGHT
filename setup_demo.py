
from auth import create_citizen_account, create_official_account

accounts = [
    ("citizen.demo@binsight.test", "CitizenDemo123", "Citizen Demo"),
    ("official.demo@binsight.test", "OfficialDemo123", "Official Demo"),
]

for email, password, name in accounts:
    print(f"\nSetting up {email}")

    # Check whether your auth.py accepts these arguments.
    # If it doesn't, stop and share the error with me.
    if email.startswith("citizen."):
        result = create_citizen_account(name, email, password)
    else:
        result = create_official_account(name, email, password)

    print(result)
