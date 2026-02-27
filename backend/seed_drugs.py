"""
Drug Database Seeder — reads from pre-built seed_data.json (5,000 medicines).

The JSON was generated from the ML training datasets locally.
It lives in backend/ so it deploys with the app (unlike ml/data/ which is gitignored).
"""
import json
import os
import sys

# Support both standalone execution and module import
try:
    from .database import SessionLocal
    from .models import DrugInformation
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from database import SessionLocal
    from models import DrugInformation

SEED_JSON = os.path.join(os.path.dirname(__file__), "seed_data.json")


def seed_db():
    print("=" * 60)
    print("  AuthChecker Drug Database Seeder")
    print("=" * 60)

    if not os.path.exists(SEED_JSON):
        print(f"[Seed] seed_data.json not found at {SEED_JSON}, skipping.")
        return

    db = SessionLocal()
    try:
        existing = db.query(DrugInformation).count()
        if existing > 0:
            print(f"\n[Seed] Database already has {existing} records. Skipping seed.")
            return

        with open(SEED_JSON, "r", encoding="utf-8") as f:
            entries = json.load(f)

        print(f"[Seed] Loaded {len(entries)} medicines from seed_data.json")

        BATCH_SIZE = 500
        total_inserted = 0

        for i in range(0, len(entries), BATCH_SIZE):
            batch = entries[i:i + BATCH_SIZE]
            for drug_data in batch:
                db.add(DrugInformation(**drug_data))
            db.commit()
            total_inserted += len(batch)
            if total_inserted % 1000 == 0 or total_inserted == len(entries):
                print(f"  Inserted {total_inserted}/{len(entries)}...")

        print(f"\n[Seed] Successfully seeded {total_inserted} medicine records!")
        print("=" * 60)

    except Exception as e:
        db.rollback()
        print(f"\n[Seed ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    seed_db()
