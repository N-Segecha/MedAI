# 📁 database.py — SQLAlchemy engine + model schemas for MedAI
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import os
from dotenv import load_dotenv

# 🌍 Load environment variables
load_dotenv()

DB_USER = os.getenv("POSTGRES_USER", "nsegecha_admin")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "123456789")
DB_NAME = os.getenv("POSTGRES_DB", "medical_db")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 🛠 SQLAlchemy setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 📄 DiagnosisRecord – stores inference results
class DiagnosisRecord(Base):
    __tablename__ = "diagnosis_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=True)
    input_type = Column(String)  # e.g., 'image', 'prompt', 'csv'
    input_source = Column(String)  # image path, text prompt, csv filename
    predicted_conditions = Column(Text)  # JSON string or flat text
    model_used = Column(String)  # Agent name or config key
    severity = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    feedback = relationship("UserFeedback", back_populates="diagnosis", uselist=True)
    aid_sessions = relationship("FirstAidSession", back_populates="diagnosis", uselist=True)

# 🗣️ UserFeedback – links to DiagnosisRecord
class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnosis_records.id"))
    feedback_text = Column(Text)
    satisfaction_rating = Column(Float)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())

    diagnosis = relationship("DiagnosisRecord", back_populates="feedback")

# 🚨 FirstAidSession – links to triage outcome
class FirstAidSession(Base):
    __tablename__ = "first_aid_sessions"

    id = Column(Integer, primary_key=True, index=True)
    diagnosis_id = Column(Integer, ForeignKey("diagnosis_records.id"))
    condition_tag = Column(String)
    guidance_steps = Column(Text)  # store as JSON string if needed
    session_start = Column(DateTime(timezone=True), server_default=func.now())

    diagnosis = relationship("DiagnosisRecord", back_populates="aid_sessions")
