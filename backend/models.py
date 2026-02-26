"""SQLAlchemy ORM models for AuthChecker.

Phase 0: Users, ValidMedicine, ScanHistory (original)
Phase 2: EmailVerification, PasswordReset, UserSession,
         DrugInformation, CounterfeitReport (new)
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey,
    Boolean, Text
)
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from .database import Base


class ScanStatus(str, enum.Enum):
    AUTHENTIC = "AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    FAKE = "FAKE"


# ─────────────────────────────────────────────
# Phase 0 — Core Tables
# ─────────────────────────────────────────────

class Users(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Phase 2.1 additions
    is_verified = Column(Boolean, default=False)
    avatar_url = Column(String, nullable=True)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    scans = relationship("ScanHistory", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    reports = relationship("CounterfeitReport", back_populates="user")


class ValidMedicine(Base):
    __tablename__ = "valid_medicines"
    id = Column(Integer, primary_key=True, index=True)
    brand_name = Column(String, index=True)
    manufacturer = Column(String)
    batch_number = Column(String, unique=True, index=True)
    expiry_date = Column(DateTime)
    mrp = Column(Float)
    packaging_hash = Column(String)
    serial_number = Column(String, unique=True, index=True)
    scan_count = Column(Integer, default=0)
    last_scanned_at = Column(DateTime, nullable=True)


class ScanHistory(Base):
    __tablename__ = "scan_history"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    scanned_batch_number = Column(String)
    authenticity_score = Column(Float)
    status = Column(String)
    image_path = Column(String)
    scanned_at = Column(DateTime, default=datetime.utcnow)

    # Phase 2.3 additions
    image_thumbnail_url = Column(String, nullable=True)
    ml_confidence = Column(Float, nullable=True)
    ml_model_version = Column(String, default="v1.0")
    result_breakdown_json = Column(Text, nullable=True)
    blockchain_tx_hash = Column(String, nullable=True)
    nfc_verified = Column(Boolean, default=False)
    deleted_at = Column(DateTime, nullable=True)  # Soft delete

    # Relationships
    user = relationship("Users", back_populates="scans")
    reports = relationship("CounterfeitReport", back_populates="scan")


# ─────────────────────────────────────────────
# Phase 2.1 — Email Verification & Password Reset
# ─────────────────────────────────────────────

class EmailVerification(Base):
    __tablename__ = "email_verifications"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    otp_code = Column(String(6), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PasswordReset(Base):
    __tablename__ = "password_resets"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    reset_token = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# Phase 2.2 — Refresh Token Rotation
# ─────────────────────────────────────────────

class UserSession(Base):
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    refresh_token = Column(String, unique=True, index=True, nullable=False)
    device_info = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("Users", back_populates="sessions")


# ─────────────────────────────────────────────
# Phase 2.5 — Drug Information
# ─────────────────────────────────────────────

class DrugInformation(Base):
    __tablename__ = "drug_information"
    id = Column(Integer, primary_key=True)
    brand_name = Column(String, index=True, nullable=False)
    generic_name = Column(String, nullable=True)
    composition = Column(Text, nullable=True)
    usage = Column(Text, nullable=True)
    side_effects = Column(Text, nullable=True)
    manufacturer = Column(String, nullable=True)
    category = Column(String, nullable=True)


# ─────────────────────────────────────────────
# Phase 2.6 — Counterfeit Reporting
# ─────────────────────────────────────────────

class CounterfeitReport(Base):
    __tablename__ = "counterfeit_reports"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scan_id = Column(Integer, ForeignKey("scan_history.id"), nullable=False)
    description = Column(Text, nullable=True)
    pharmacy_name = Column(String, nullable=True)
    pharmacy_location = Column(String, nullable=True)
    geo_lat = Column(Float, nullable=True)
    geo_long = Column(Float, nullable=True)
    status = Column(String, default="pending")  # pending / reviewed / resolved
    reported_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("Users", back_populates="reports")
    scan = relationship("ScanHistory", back_populates="reports")