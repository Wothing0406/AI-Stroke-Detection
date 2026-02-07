from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import audio_processing
import feature_extraction
import ml_model
import report_generator
import uuid

app = FastAPI()

# Enable CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
REPORTS_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Stroke Detection AI API is running"}

@app.get("/status")
def get_status():
    """
    Return the status of the system and its components.
    """
    return {
        "status": "online",
        "features": [
            {"name": "Xử lý âm thanh (Audio Processing)", "status": "Ready", "percent": 100},
            {"name": "Trích xuất đặc trưng (MFCC/Jitter)", "status": "Ready", "percent": 100},
            {"name": "Mô hình AI (SVM Model)", "status": "Ready", "percent": 100},
            {"name": "API Kết nối", "status": "Ready", "percent": 100}
        ],
        "completion_rate": 100
    }

@app.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    name: str = Form(...),
    dob: str = Form(...),
    cccd: str = Form(...)
):
    """
    Analyze uploaded audio file for potential stroke indicators.
    Also generates a medical report PDF.
    """
    # Generate unique ID for this session
    session_id = str(uuid.uuid4())
    temp_file_path = f"{UPLOAD_DIR}/{session_id}_{file.filename}"
    report_filename = f"Report_{session_id}.pdf"
    report_path = f"{REPORTS_DIR}/{report_filename}"
    
    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 1. Load and Process Audio
        y, sr = audio_processing.load_audio(temp_file_path)
        if y is None:
            raise HTTPException(status_code=400, detail="Could not process audio file")
            
        # 2. Extract Features
        features = feature_extraction.extract_features(y, sr)
        if features is None:
             raise HTTPException(status_code=400, detail="Could not extract features from audio")

        # 3. Predict Risk
        risk_label, confidence = ml_model.predict_risk(features)
        
        # 4. Generate Report
        patient_info = {"name": name, "dob": dob, "cccd": cccd}
        result = {
            "risk_assessment": risk_label,
            "confidence_score": float(confidence),
            "details": "Analysis based on MFCC, Spectral Centroid, and ZCR."
        }
        
        report_generator.generate_medical_report(patient_info, result, temp_file_path, report_path)
        
        return {
            "filename": file.filename,
            "risk_assessment": risk_label,
            "confidence_score": float(confidence),
            "details": result['details'],
            "report_url": f"/report/{report_filename}",
            "patient_info": patient_info
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    # Note: We keep the audio file to generate waveform in report, but we could clean up later.

@app.get("/report/{filename}")
async def download_report(filename: str):
    file_path = f"{REPORTS_DIR}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="Report not found")


if __name__ == "__main__":
    import uvicorn
    # Initial model training if needed
    if not os.path.exists(ml_model.MODEL_PATH):
        ml_model.train_dummy_model()
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
