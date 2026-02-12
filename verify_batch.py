from backend.database import SessionLocal
from backend.models import ValidMedicine

db = SessionLocal()
batch = "EA25049"
exists = db.query(ValidMedicine).filter(ValidMedicine.batch_number == batch).first()

if exists:
    print(f"SUCCESS: Batch {batch} found in DB! Brand: {exists.brand_name}")
else:
    print(f"FAILURE: Batch {batch} NOT found in DB.")
db.close()
