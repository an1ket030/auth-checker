import sys
from backend.database import SessionLocal
from backend.models import Users

try:
    db = SessionLocal()
    user = db.query(Users).filter(Users.username == "demo").first()
    if user:
        print(f"USER: {user.username}")
        print(f"HASH: {user.password_hash!r}")
    else:
        print("User 'demo' NOT FOUND")
except Exception as e:
    print(f"ERROR: {e}")
