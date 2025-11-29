from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# PRODUCTION: Use environment variables for credentials. DO NOT hardcode passwords.
# Example URL: postgresql://user:password@localhost/authchecker_db
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:anikeet030@localhost/authchecker_db")

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