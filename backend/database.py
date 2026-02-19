from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from dotenv import load_dotenv

# Load .env from the same directory as this file
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# PRODUCTION: Use environment variables for credentials. DO NOT hardcode passwords.
# Example URL: postgresql://user:password@localhost/authchecker_db
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Clean up the URL string (handle accidental quotes or whitespace)
SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.strip().strip("'").strip('"')

# --- DEBUG --- 
# Print the connection URL (masking password) to see exactly what Render is using
try:
    from sqlalchemy.engine.url import make_url
    url_obj = make_url(SQLALCHEMY_DATABASE_URL)
    print(f"\n[DEBUG] Connecting to Database:")
    print(f"  > Host: {url_obj.host}")
    print(f"  > Port: {url_obj.port}")
    print(f"  > User: {url_obj.username}")
    print(f"  > DB  : {url_obj.database}")
    print(f"  > Full (Masked): {url_obj.__repr__()}\n")
except Exception as e:
    print(f"[DEBUG] Could not parse URL for debug printing: {e}")
# -------------

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency injection for API routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()