from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base

class ScanStatus(str, enum.Enum):
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    FAKE = "FAKE"

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class ValidMedicine(Base):
    __tablename__ = "valid_medicines"
    id = Column(Integer, primary_key=True, index=True)
    brand_name = Column(String, index=True)
    manufacturer = Column(String)
    batch_number = Column(String, unique=True, index=True)
    expiry_date = Column(DateTime)
    mrp = Column(Float)
    packaging_hash = Column(String)
    serial_number = Column(String, unique=True, index=True) # Anti-replay: unique per unit
    
    # -- Anti-Cloning / Supply Chain --
    scan_count = Column(Integer, default=0)
    last_scanned_at = Column(DateTime, nullable=True)

class ScanHistory(Base):
    __tablename__ = "scan_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id")) # Linked to User
    scanned_batch_number = Column(String)
    authenticity_score = Column(Float)
    status = Column(String)
    image_path = Column(String)
    scanned_at = Column(DateTime, default=datetime.utcnow)