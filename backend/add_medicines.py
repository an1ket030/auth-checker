from datetime import datetime
from decimal import Decimal
import os

# Use the same imports as your app so config/URL are reused
from backend.database import SessionLocal
from backend.models import ValidMedicine

EXAMPLE_PACKAGING_IMAGE = "/mnt/data/600b983a-d74a-47c9-8c19-3bdac88ed83d.png"

MEDICINES_TO_ADD = [
    {
        "brand_name": "Crocin Advance",
        "manufacturer": "GSK Pharma",
        "batch_number": "EA25049",
        "expiry_date": datetime(2027, 6, 1),
        "mrp": 95.0,
        "packaging_hash": EXAMPLE_PACKAGING_IMAGE,
    },
    {
        "brand_name": "Cyclopam 250",
        "manufacturer": "Indoco Remedies",
        "batch_number": "25070246",
        "expiry_date": datetime(2026, 10, 1),
        "mrp": 63.40,
        "packaging_hash": EXAMPLE_PACKAGING_IMAGE,
    },
    
    {
        "brand_name": "Ondet-4",
        "manufacturer": "Intas Pharmaceuticals",
        "batch_number": "INK24K005",
        "expiry_date": datetime(2026, 8, 1),
        "mrp": 57.51,
        "packaging_hash": EXAMPLE_PACKAGING_IMAGE,
    },
    {
        "brand_name": "MEFTAL-SPAS",
        "manufacturer": "Blue Cross Laboratories",
        "batch_number": "YMS2584",
        "expiry_date": datetime(2028, 2, 1),
        "mrp": 52.00,
        "packaging_hash": EXAMPLE_PACKAGING_IMAGE,
    },
    {
        "brand_name": "Lopravik",
        "manufacturer": "Vikram Labratories",
        "batch_number": "25238",
        "expiry_date": datetime(2027, 7, 1),
        "mrp": 63.40,
        "packaging_hash": EXAMPLE_PACKAGING_IMAGE,
    },
    {
        "brand_name": "Azithro 500",
        "manufacturer": "Cipla Ltd",
        "batch_number": "AZ500X12",
        "expiry_date": datetime(2026, 11, 1),
        "mrp": 120.0,
        "packaging_hash": EXAMPLE_PACKAGING_IMAGE,
    },
    {
        "brand_name": "Paracetamol 650",
        "manufacturer": "MedLife Labs",
        "batch_number": "P650-2024-01",
        "expiry_date": datetime(2026, 1, 1),
        "mrp": 40.0,
        "packaging_hash": EXAMPLE_PACKAGING_IMAGE,
    },
]

def main():
    db = SessionLocal()
    added = 0
    try:
        for m in MEDICINES_TO_ADD:
            # Check if batch_number exists (batch_number is unique in your model)
            existing = db.query(ValidMedicine).filter(ValidMedicine.batch_number == m["batch_number"]).first()
            if existing:
                print(f"SKIP: batch {m['batch_number']} already exists (id={existing.id})")
                continue
            record = ValidMedicine(
                brand_name=m["brand_name"],
                manufacturer=m["manufacturer"],
                batch_number=m["batch_number"],
                expiry_date=m["expiry_date"],
                mrp=float(m["mrp"]),
                packaging_hash=m.get("packaging_hash")
            )
            db.add(record)
            added += 1
        db.commit()
        print(f"Done. Added {added} medicine(s).")
    except Exception as e:
        db.rollback()
        print("Error:", e)
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()




#---------------------------------------------------------
    # INSERT INTO valid_medicines (brand_name, manufacturer, batch_number, expiry_date, mrp, packaging_hash)
    # VALUES
    # ('Crocin Advance', 'GSK Pharma', 'EA25049', '2027-06-01', 95.0, '/mnt/data/600b983a-d74a-47c9-8c19-3bdac88ed83d.png'),
    # ('Crocin Advance', 'GSK Pharma', 'EA25050', '2027-07-01', 95.0, '/mnt/data/600b983a-d74a-47c9-8c19-3bdac88ed83d.png'),
    # ('Azithro 500', 'Cipla Ltd', 'AZ500X12', '2026-11-01', 120.0, '/mnt/data/600b983a-d74a-47c9-8c19-3bdac88ed83d.png'),
    # ('Paracetamol 650', 'MedLife Labs', 'P650-2024-01', '2026-01-01', 40.0, '/mnt/data/600b983a-d74a-47c9-8c19-3bdac88ed83d.png');
#------------------------------------------------------------