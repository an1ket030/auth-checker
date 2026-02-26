import json
import os
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import DrugInformation, Base

# Medicine data tailored to the Indian market
MEDICINES_DATA = [
    {
        "brand_name": "Dolo 650",
        "generic_name": "Paracetamol",
        "composition": "Paracetamol (Acetaminophen) 650mg",
        "usage": "Pain relief, fever reduction",
        "side_effects": "Nausea, allergic reactions, liver damage in overdose",
        "manufacturer": "Micro Labs Ltd",
        "category": "Analgesic & Antipyretic"
    },
    {
        "brand_name": "Augmentin 625 Duo",
        "generic_name": "Amoxicillin + Clavulanic Acid",
        "composition": "Amoxicillin 500mg, Clavulanic Acid 125mg",
        "usage": "Bacterial infections (respiratory, ear, skin)",
        "side_effects": "Diarrhea, nausea, skin rashes",
        "manufacturer": "GlaxoSmithKline Pharmaceuticals Ltd",
        "category": "Antibiotic"
    },
    {
        "brand_name": "Pan 40",
        "generic_name": "Pantoprazole",
        "composition": "Pantoprazole 40mg",
        "usage": "Acid reflux, peptic ulcer disease, GERD",
        "side_effects": "Headache, diarrhea, stomach pain",
        "manufacturer": "Alkem Laboratories Ltd",
        "category": "Antacid (Proton Pump Inhibitor)"
    },
    {
        "brand_name": "Telma 40",
        "generic_name": "Telmisartan",
        "composition": "Telmisartan 40mg",
        "usage": "Hypertension (high blood pressure), heart failure prevention",
        "side_effects": "Dizziness, back pain, sinus infection",
        "manufacturer": "Glenmark Pharmaceuticals Ltd",
        "category": "Antihypertensive"
    },
    {
        "brand_name": "Thyronorm 50",
        "generic_name": "Thyroxine",
        "composition": "Levothyroxine 50mcg",
        "usage": "Hypothyroidism (underactive thyroid)",
        "side_effects": "Weight loss, rapid heart rate, sweating",
        "manufacturer": "Abbott India Ltd",
        "category": "Thyroid Hormone"
    },
    {
        "brand_name": "Allegra 120",
        "generic_name": "Fexofenadine",
        "composition": "Fexofenadine 120mg",
        "usage": "Allergies, hay fever, hives",
        "side_effects": "Headache, drowsiness, dry mouth",
        "manufacturer": "Sanofi India Ltd",
        "category": "Antihistamine"
    },
    {
        "brand_name": "Metrogyl 400",
        "generic_name": "Metronidazole",
        "composition": "Metronidazole 400mg",
        "usage": "Parasitic and bacterial infections, amoebiasis",
        "side_effects": "Metallic taste, nausea, dry mouth",
        "manufacturer": "J.B. Chemicals & Pharmaceuticals Ltd",
        "category": "Antimicrobial"
    },
    {
        "brand_name": "Ascoril LS",
        "generic_name": "Ambroxol + Levosalbutamol + Guaifenesin",
        "composition": "Levosalbutamol 1mg, Ambroxol 30mg, Guaifenesin 50mg / 5ml",
        "usage": "Cough with mucus, asthma, bronchitis",
        "side_effects": "Tremors, increased heart rate, dizziness",
        "manufacturer": "Glenmark Pharmaceuticals Ltd",
        "category": "Cough Syrup/Expectorant"
    },
    {
        "brand_name": "Amlokind-AT",
        "generic_name": "Amlodipine + Atenolol",
        "composition": "Amlodipine 5mg, Atenolol 50mg",
        "usage": "Hypertension, angina",
        "side_effects": "Slow heart rate, fatigue, cold extremities",
        "manufacturer": "Mankind Pharma Ltd",
        "category": "Antihypertensive"
    },
    {
        "brand_name": "Lipicard 160",
        "generic_name": "Fenofibrate",
        "composition": "Fenofibrate 160mg",
        "usage": "High cholesterol, high triglycerides",
        "side_effects": "Stomach upset, liver enzyme elevations",
        "manufacturer": "USV Pvt Ltd",
        "category": "Lipid-Lowering Agent"
    },
    {
        "brand_name": "Calpol 500",
        "generic_name": "Paracetamol",
        "composition": "Paracetamol 500mg",
        "usage": "Mild pain relief, fever",
        "side_effects": "Rarely liver toxicity if overused",
        "manufacturer": "GlaxoSmithKline Pharmaceuticals Ltd",
        "category": "Analgesic & Antipyretic"
    },
    {
        "brand_name": "Montair LC",
        "generic_name": "Montelukast + Levocetirizine",
        "composition": "Montelukast 10mg, Levocetirizine 5mg",
        "usage": "Allergic rhinitis, asthma",
        "side_effects": "Sleepiness, fatigue, dry mouth",
        "manufacturer": "Cipla Ltd",
        "category": "Antiallergic"
    },
    {
        "brand_name": "Ondem 4",
        "generic_name": "Ondansetron",
        "composition": "Ondansetron 4mg",
        "usage": "Nausea and vomiting",
        "side_effects": "Headache, constipation, sensation of warmth",
        "manufacturer": "Alkem Laboratories Ltd",
        "category": "Antiemetic"
    },
    {
        "brand_name": "Ecosprin 75",
        "generic_name": "Aspirin",
        "composition": "Aspirin 75mg",
        "usage": "Prevention of heart attacks and strokes, blood thinner",
        "side_effects": "Bleeding, indigestion, heartburn",
        "manufacturer": "USV Pvt Ltd",
        "category": "Antiplatelet / Blood Thinner"
    },
    {
        "brand_name": "Shelcal 500",
        "generic_name": "Calcium Carbonate + Vitamin D3",
        "composition": "Calcium Carbonate 500mg, Vitamin D3 250 IU",
        "usage": "Calcium deficiency, osteoporosis",
        "side_effects": "Constipation, bloating",
        "manufacturer": "Torrent Pharmaceuticals Ltd",
        "category": "Calcium Supplement"
    }
]

def seed_db():
    print("Starting database seed for Drug Information...")
    db: Session = SessionLocal()
    try:
        # Check if already seeded
        existing_count = db.query(DrugInformation).count()
        if existing_count > 0:
            print(f"Database already contains {existing_count} drug records. Skipping seed.")
            return

        for drug in MEDICINES_DATA:
            new_drug = DrugInformation(**drug)
            db.add(new_drug)
        
        db.commit()
        print(f"Successfully seeded {len(MEDICINES_DATA)} medicine records.")

    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_db()
