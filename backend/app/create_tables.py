from backend.app.database import Base, engine
from backend.app.database import DiagnosisRecord, UserFeedback, FirstAidSession

Base.metadata.create_all(bind=engine)
print("✅ Tables created successfully.")
