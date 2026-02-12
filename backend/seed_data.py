import sys
import os
import random
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from backend.database import SessionLocal, engine, Base
from backend.models import ValidMedicine, Users

# Ensure we can import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def reset_db():
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Creating all tables...")
    Base.metadata.create_all(bind=engine)

def seed_data():
    db = SessionLocal()
    try:
        # Realistic Medicine Data
        # Source: Common medicines in India/US
        medicines = [
            {"brand": "Dolo 650", "manuf": "Micro Labs Ltd", "mrp": 30.0},
            {"brand": "Crocin Advance", "manuf": "GlaxoSmithKline", "mrp": 20.0},
            {"brand": "Pan 40", "manuf": "Alkem Laboratories", "mrp": 155.0},
            {"brand": "Azithral 500", "manuf": "Alembic Pharmaceuticals", "mrp": 119.0},
            {"brand": "Shelcal 500", "manuf": "Torrent Pharmaceuticals", "mrp": 110.0},
            {"brand": "Becosules Z", "manuf": "Pfizer Ltd", "mrp": 45.0},
            {"brand": "Allegra 120", "manuf": "Sanofi India", "mrp": 210.0},
            {"brand": "Telma 40", "manuf": "Glenmark Pharmaceuticals", "mrp": 240.0},
            {"brand": "Augmentin 625", "manuf": "GlaxoSmithKline", "mrp": 223.0},
            {"brand": "Ascoril LS", "manuf": "Glenmark Pharmaceuticals", "mrp": 115.0},
        ]

        print("Seeding ValidMedicine...")
        for med in medicines:
            # Generate 5-10 batches per medicine
            for _ in range(random.randint(5, 10)):
                # Simulate realistic batch number: mostly alphanumeric, 6-10 chars
                batch_base = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
                batch_num = "".join(random.choices("0123456789", k=6))
                batch_full = f"{batch_base}{batch_num}"
                
                # Expiry 1-3 years in future
                exp_date = datetime.now() + timedelta(days=random.randint(365, 1000))
                
                # Create multiple units (Serial Numbers) for this batch
                # To simulate "Golden Record" where we know the valid batch
                # We also create a few specific serial numbers that match our demo labels if any
                
                vm = ValidMedicine(
                    brand_name=med["brand"],
                    manufacturer=med["manuf"],
                    batch_number=batch_full,
                    expiry_date=exp_date,
                    mrp=med["mrp"],
                    packaging_hash="mock_hash",
                    serial_number=f"SN{random.randint(10000000, 99999999)}" # Unique Serial Number
                )
                db.add(vm)
        
        # Add a specific known batch for testing demo
        # Batch: B452202, Brand: Dolo 650
        demo_med = ValidMedicine(
            brand_name="Dolo 650",
            manufacturer="Micro Labs Ltd",
            batch_number="B452202",
            expiry_date=datetime(2025, 12, 31),
            mrp=30.0,
            packaging_hash="demo_hash",
            serial_number="SN12345678"
        )
        db.add(demo_med)

        # Add User Provided Batch for Testing
        # Batch: EA25049, Brand: Crocin Advance
        user_test_med = ValidMedicine(
            brand_name="Crocin Advance",
            manufacturer="GlaxoSmithKline",
            batch_number="EA25049",
            expiry_date=datetime(2025, 12, 31),
            mrp=20.0,
            packaging_hash="user_test_hash",
            serial_number="SN99999999"
        )
        db.add(user_test_med)
        
        # Add a demo user
        print("Seeding Users...")
        # from passlib.context import CryptContext
        # pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
        from passlib.hash import pbkdf2_sha256
        
        user = Users(
            username="demo", 
            email="demo@example.com",
            password_hash=pbkdf2_sha256.hash("password123")
        )
        db.add(user)

        db.commit()
        print("Seeding complete!")
        print(f"Inserted {db.query(ValidMedicine).count()} medicine records.")
        print(f"Inserted {db.query(Users).count()} users.")

    except Exception as e:
        print(f"Error seeding data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    reset_db()
    seed_data()
