"""
Comprehensive Drug Database Seeder
Reads from the project's ML training datasets to populate the drug_information table.

Sources:
  1. ml/data/raw/indian_pharmaceutical_products_clean.csv  (253k rows)
  2. ml/data/raw/medicine_data.csv                         (195k rows)
"""
import csv
import os
import ast
import sys

# Support both standalone execution and module import
try:
    from .database import SessionLocal
    from .models import DrugInformation
except ImportError:
    # Standalone execution from backend/ directory
    sys.path.insert(0, os.path.dirname(__file__))
    from database import SessionLocal
    from models import DrugInformation

# Paths (relative to project root)
BASE = os.path.dirname(os.path.dirname(__file__))
PHARMA_CSV = os.path.join(BASE, "ml", "data", "raw", "indian_pharmaceutical_products_clean.csv")
MEDICINE_CSV = os.path.join(BASE, "ml", "data", "raw", "medicine_data.csv")


def parse_pharma_csv():
    """
    Parse indian_pharmaceutical_products_clean.csv
    Columns: product_id, brand_name, manufacturer, price_inr, is_discontinued,
             dosage_form, pack_size, pack_unit, num_active_ingredients,
             primary_ingredient, primary_strength, active_ingredients,
             therapeutic_class, packaging_raw, manufacturer_raw
    """
    drugs = {}
    if not os.path.exists(PHARMA_CSV):
        print(f"[Seed] Pharma CSV not found at {PHARMA_CSV}, skipping.")
        return drugs

    with open(PHARMA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip discontinued
            if row.get("is_discontinued", "").strip().lower() == "true":
                continue

            brand = row.get("brand_name", "").strip()
            if not brand:
                continue

            # Deduplicate by normalized brand name
            key = brand.lower()
            if key in drugs:
                continue

            # Parse active_ingredients list
            composition = ""
            try:
                ingredients = ast.literal_eval(row.get("active_ingredients", "[]"))
                if isinstance(ingredients, list):
                    composition = " + ".join(
                        ing.get("full_description", ing.get("name", ""))
                        for ing in ingredients
                    )
            except:
                composition = row.get("primary_ingredient", "")
                strength = row.get("primary_strength", "")
                if strength:
                    composition += f" ({strength})"

            category = row.get("therapeutic_class", "").strip().title()
            manufacturer = row.get("manufacturer", "").strip()
            generic = row.get("primary_ingredient", "").strip()

            drugs[key] = {
                "brand_name": brand,
                "generic_name": generic,
                "composition": composition,
                "usage": None,  # Will be enriched from medicine_data.csv
                "side_effects": None,
                "manufacturer": manufacturer,
                "category": category if category else None,
            }

    print(f"[Seed] Parsed {len(drugs)} unique medicines from pharma CSV.")
    return drugs


def enrich_from_medicine_csv(drugs):
    """
    Enrich with descriptions and side effects from medicine_data.csv
    Columns: sub_category, product_name, salt_composition, product_price,
             product_manufactured, medicine_desc, side_effects, drug_interactions
    """
    if not os.path.exists(MEDICINE_CSV):
        print(f"[Seed] Medicine CSV not found at {MEDICINE_CSV}, skipping enrichment.")
        return drugs

    enriched_count = 0
    new_count = 0

    with open(MEDICINE_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product = row.get("product_name", "").strip()
            if not product:
                continue

            key = product.lower()

            if key in drugs:
                # Enrich existing entry
                if not drugs[key]["usage"] and row.get("medicine_desc"):
                    desc = row["medicine_desc"].strip()
                    # Truncate very long descriptions
                    drugs[key]["usage"] = desc[:500] if len(desc) > 500 else desc
                    enriched_count += 1
                if not drugs[key]["side_effects"] and row.get("side_effects"):
                    se = row["side_effects"].strip()
                    drugs[key]["side_effects"] = se[:500] if len(se) > 500 else se
            else:
                # Add as new entry
                salt = row.get("salt_composition", "").strip()
                desc = row.get("medicine_desc", "").strip()
                se = row.get("side_effects", "").strip()
                mfr = row.get("product_manufactured", "").strip()
                cat = row.get("sub_category", "").strip().title()

                drugs[key] = {
                    "brand_name": product,
                    "generic_name": salt[:200] if salt else None,
                    "composition": salt[:300] if salt else None,
                    "usage": desc[:500] if desc else None,
                    "side_effects": se[:500] if se else None,
                    "manufacturer": mfr if mfr else None,
                    "category": cat if cat else None,
                }
                new_count += 1

    print(f"[Seed] Enriched {enriched_count} entries from medicine CSV, added {new_count} new entries.")
    return drugs


def seed_db():
    print("=" * 60)
    print("  AuthChecker Drug Database Seeder")
    print("=" * 60)

    db: Session = SessionLocal()
    try:
        existing = db.query(DrugInformation).count()
        if existing > 0:
            print(f"\n[Seed] Clearing existing {existing} records to reseed...")
            db.query(DrugInformation).delete()
            db.commit()

        # Step 1: Parse pharma CSV
        drugs = parse_pharma_csv()

        # Step 2: Enrich from medicine CSV
        drugs = enrich_from_medicine_csv(drugs)

        # Step 3: Filter out entries with no useful data
        filtered = {
            k: v for k, v in drugs.items()
            if v["brand_name"] and (v["composition"] or v["generic_name"])
        }
        print(f"\n[Seed] Total unique medicines after filtering: {len(filtered)}")

        # Step 3.5: Limit to top ~5000 most complete entries (for Render free tier performance)
        MAX_SEED = 5000

        def completeness_score(drug):
            """Score from 0-5 based on how many fields are filled."""
            score = 0
            if drug.get("composition"): score += 1
            if drug.get("usage"): score += 1
            if drug.get("side_effects"): score += 1
            if drug.get("manufacturer"): score += 1
            if drug.get("category"): score += 1
            return score

        entries = sorted(filtered.values(), key=completeness_score, reverse=True)
        if len(entries) > MAX_SEED:
            print(f"[Seed] Limiting to top {MAX_SEED} most complete entries (from {len(entries)}).")
            entries = entries[:MAX_SEED]

        # Step 4: Insert in batches
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
