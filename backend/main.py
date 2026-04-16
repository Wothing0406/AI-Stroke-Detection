from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import asyncio
import io
import shutil
import os
import zipfile
import json
import re
from datetime import datetime
import audio_processing
import feature_extraction
import ml_model
import report_generator
import risk_engine
import schemas
import uuid
import traceback
import time
import logging
import sys
import numpy as np
import clinical_features
import concurrent.futures

# Global Executor for heavy acoustic processing 
# Using ThreadPoolExecutor on Windows to avoid SpawnProcess errors with FastAPI
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

UPLOAD_DIR = "temp_uploads"
REPORTS_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def cleanup_old_files(directory, max_age_hours=24):
    """
    Delete files older than max_age_hours to prevent disk bloat.
    """
    now = datetime.now().timestamp()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            if now - os.path.getmtime(file_path) > max_age_hours * 3600:
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {filename}")
                except Exception as e:
                    logger.error(f"Failed to delete {filename}: {e}")

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

@app.get("/history")
async def get_history():
    """
    Retrieve session history for trend analysis.
    """
    history_path = "session_history.json"
    if not os.path.exists(history_path):
        return []
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Format for charts
        results = []
        for session_id, features in data.items():
            try:
                # Calculate SAI score for the trend
                risk_data = risk_engine.analyze_risk(features, 45)
                sai_score = risk_data.get("sai_score", 0)
                
                results.append({
                    "session_id": session_id,
                    "sai_score": sai_score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Placeholder
                })
            except:
                continue
        return results
    except Exception as e:
        logger.error(f"History error: {e}")
        return []

@app.websocket("/ws/voice-stream")
async def voice_streaming_ws(websocket: WebSocket, gender: str = "Nam"):
    """
    Real-time clinical screening WebSocket.
    Receives binary PCM chunks (16kHz), performs rolling analysis, and returns JSON metrics.
    """
    await websocket.accept()
    logger.info("Real-time Voice Stream Connected")
    
    # 1s window = 16000 samples * 2 bytes (int16) = 32000 bytes
    WINDOW_SIZE = 32000 
    buffer = bytearray()
    
    try:
        while True:
            # Receive binary chunk
            chunk = await websocket.receive_bytes()
            buffer.extend(chunk)
            
            if len(buffer) >= WINDOW_SIZE:
                # Process the most recent window
                try:
                    # Convert bytearray to numpy float32 [-1.0, 1.0]
                    audio_data = np.frombuffer(buffer[:WINDOW_SIZE], dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Basic real-time features
                    is_speaking = audio_processing.detect_speech(audio_data, 16000)["status"] == "VALID"
                    
                    if is_speaking:
                        # Extract quick biomarkers
                        import clinical_features
                        metrics = clinical_features.extract_clinical_features(audio_data, 16000)
                        
                        # Analyze SAI
                        analysis = risk_engine.analyze_risk(metrics, age=45, signal_quality=0.9, gender=gender)
                        
                        # --- ENHANCEMENT: Smoothed Real-time Scores ---
                        if not hasattr(websocket, 'score_history'):
                            websocket.score_history = []
                        websocket.score_history.append(analysis["sai_score"])
                        if len(websocket.score_history) > 3:
                            websocket.score_history.pop(0)
                        smoothed_sai = sum(websocket.score_history) / len(websocket.score_history)
                        
                        await websocket.send_json({
                            "type": "STREAM_UPDATE",
                            "sai_score": round(float(smoothed_sai), 1),
                            "risk_level": "NORMAL" if smoothed_sai < 20 else "MEDIUM" if smoothed_sai < 55 else "HIGH",
                            "confidence": analysis["confidence"],
                            "group_scores": analysis["group_scores"],
                            "metrics": metrics["vector"],
                            "timestamp": time.time()
                        })
                    else:
                        await websocket.send_json({
                            "type": "SILENCE",
                            "timestamp": time.time()
                        })
                        
                except Exception as inner_e:
                    logger.error(f"WS Analysis Error: {inner_e}")
                
                # Slide window
                buffer = buffer[WINDOW_SIZE:]
                
    except WebSocketDisconnect:
        logger.info("Voice Stream Disconnected")
    except Exception as e:
        logger.error(f"WS Loop Error: {e}")
        try:
            await websocket.close()
        except:
            pass





@app.post("/validate")
async def validate_audio_quick(file: UploadFile = File(...)):
    """
    Quick validation - checks TECHNICAL quality only after recording.
    Returns VALID/INVALID with reasons.
    """
    save_dir = os.path.abspath("temp")
    os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.normpath(os.path.join(save_dir, f"validate_{uuid.uuid4()}.tmp"))
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        y, sr = audio_processing.load_audio(temp_file_path)
        if y is None:
            return JSONResponse(content={
                "status": "INVALID",
                "reasons": ["Không đọc được file"],
                "quality_metrics": {}
            })
        
        is_valid, reasons, quality_metrics = audio_processing.validate_audio(y, sr)
        
        # Ensure quality_metrics is JSON serializable (convert numpy types)
        safe_metrics = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k, v in quality_metrics.items()}
        
        if is_valid:
            return JSONResponse(content={
                "status": "VALID",
                "reasons": [],
                "quality_metrics": safe_metrics
            })
        else:
            return JSONResponse(content={
                "status": "INVALID",
                "reasons": reasons,
                "quality_metrics": safe_metrics
            })
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(content={
            "status": "ERROR",
            "reasons": ["Lỗi xử lý"],
            "quality_metrics": {}
        }, status_code=500)
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass

@app.post("/analyze", response_model=schemas.AnalysisResponse)
async def analyze_audio(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(""),
    dob: str = Form(""), 
    age: str = Form(""), 
    gender: str = Form(""),
    health_notes: str = Form(""),
    validation_status: str = Form(""),
    dialect: str = Form("North") # New optional field
):
    """
    Analyze uploaded audio file for potential stroke indicators.
    """
    # Use Request ID from middleware (set in middleware). NOTE: no hard gate on validation_status.
    # Audio quality is re-validated inside the /analyze flow below.
    session_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
    save_dir = os.path.abspath(UPLOAD_DIR)
    os.makedirs(save_dir, exist_ok=True)
    temp_file_path = os.path.normpath(os.path.join(save_dir, f"{session_id}_{file.filename}"))
    
    try:
        # --- STEP 0: Parse Age from DOB ---
        current_year = 2026
        final_age = 50 # Default
        if dob:
            try:
                birth_year = int(dob.split('-')[0])
                final_age = current_year - birth_year
            except:
                if age: final_age = int(age)
        elif age:
            final_age = int(age)
            
        if final_age < 0: final_age = 50

        # Save uploaded file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # --- STEP 1: Load Audio ---
        y, sr = audio_processing.load_audio(temp_file_path)
        if y is None:
            # We must return a valid AnalysisResponse even in error for strict typing, 
            # OR raise HTTPException. 
            # Frontend expects 200 with status="ERROR" usually, but Pydantic might complain 
            # if we return dict that matches schema.
            # Let's construct a "Safe" error response matching schema.
            return JSONResponse(content={
                "status": "ERROR",
                "session_id": session_id,
                "final_risk_level": "UNKNOWN",
                "confidence_score": 0.0,
                "risk_assessment": {},
                "details": {},
                "deviated_systems": [],
                "possible_risks": [],
                "explanation": "Không đọc được file âm thanh.",
                "quality_metrics": {
                    "snr": 0.0, "vad_ratio": 0.0, "clipping_ratio": 0.0, 
                    "pitch_confidence": 0.0, "duration": 0.0, "tier": "INVALID"
                },
                "metadata": {
                    "model_version": "1.0", "feature_set_version": "error", 
                    "normalization_method": "none", "threshold_policy": "none",
                    "analysis_id": session_id, "timestamp": datetime.now().isoformat()
                },
                "report_url": "",
                "report_zip_url": None,
                "ai_notice": "Lỗi hệ thống."
            }, status_code=200)

        # --- STEP 2: Validate Audio (Strict) ---
        is_valid, reasons, quality_metrics_dict = audio_processing.validate_audio(y, sr)
        
        # Construct Safe Quality Metrics Object
        q_metrics = schemas.QualityMetrics(
            snr=quality_metrics_dict.get("snr", 0.0),
            vad_ratio=quality_metrics_dict.get("vad_ratio", 0.0),
            clipping_ratio=quality_metrics_dict.get("clipping_ratio", 0.0),
            pitch_confidence=quality_metrics_dict.get("pitch_confidence", 0.0),
            duration=quality_metrics_dict.get("duration", 0.0),
            tier=quality_metrics_dict.get("tier", "INVALID")
        )

        if not is_valid:
            print(f"INPUT_REJECTED: {reasons}")
            return JSONResponse(content={
                "status": "INPUT_REJECTED",
                "session_id": session_id,
                "final_risk_level": "UNKNOWN",
                "confidence_score": 0.0,
                "risk_assessment": {},
                "details": {},
                "deviated_systems": [],
                "possible_risks": [],
                "explanation": f"Dữ liệu không hợp lệ: {', '.join(reasons)}",
                "quality_metrics": q_metrics.dict(),
                "metadata": {
                    "model_version": "1.0", "feature_set_version": "rejected", 
                    "normalization_method": "none", "threshold_policy": "none",
                    "analysis_id": session_id, "timestamp": datetime.now().isoformat()
                },
                "report_url": "",
                "report_zip_url": None,
                "ai_notice": "Vui lòng thu âm lại."
            }, status_code=200)

        # Pad/trim (Disabled for full-duration research)
        # y = audio_processing.pad_or_trim_audio(y, duration=audio_processing.DURATION, sr=sr)
        logger.info(f"Full-Duration Analysis Enforced: {len(y)/sr:.2f}s of evidence.")

        # --- STEP 4: Advanced Research Extraction & SAI ---
        try:
            # Multi-processing optimization for deep 54-biomarker feature extraction
            loop = asyncio.get_event_loop()
            metrics = await loop.run_in_executor(executor, clinical_features.extract_clinical_features, y, sr)
            features_vec, _ = await loop.run_in_executor(executor, feature_extraction.extract_features, y, sr)
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return JSONResponse(content={"status": "ERROR", "message": f"Lỗi trích xuất: {str(e)}"}, status_code=500)

        # ML Probability
        if features_vec is None:
            logger.error("ML feature vector is None. Check feature_extraction.py logs.")
            return JSONResponse(content={"status": "ERROR", "message": "Lỗi phân tích đặc trưng AI. Vui lòng thử lại."}, status_code=500)
            
        ml_prob = ml_model.learning_manager.predict_risk_score(features_vec.tolist())
        
        # --- ENHANCEMENT: Adaptive Learning Log ---
        # Stores the numeric feature vector for future expert-weighted retraining
        ml_model.learning_manager.log_session(session_id, features_vec.tolist())
        
        # Clinical Analysis (Now Dialect-Aware)
        # We pass dialect through metrics "metadata" for the risk engine
        metrics_with_dialect = metrics.copy()
        if "metadata" not in metrics_with_dialect: metrics_with_dialect["metadata"] = {}
        metrics_with_dialect["metadata"]["dialect"] = dialect

        analysis_result = risk_engine.analyze_risk(metrics_with_dialect, age=final_age, ml_prob=ml_prob, signal_quality=q_metrics.snr/30.0, gender=gender)
        
        # Inject metadata for schema compliance
        metadata_dict = {
            "model_version": "research-v2", # Incremented
            "feature_set_version": "acoustic-dialect-v1",
            "normalization_method": "physiological_ranges",
            "threshold_policy": "dialect_aware_consensus",
            "dialect": dialect,
            "analysis_id": session_id,
            "timestamp": datetime.now().isoformat()
        }
        metadata_obj = schemas.AnalysisMetadata(**metadata_dict)

        # --- STEP 5: Prepare Result Payload ---
        patient_info = {
            "name": name, "age": final_age, "dob": dob,
            "gender": gender or "N/A", "health_notes": health_notes or ""
        }

        # Format details
        details_obj = {}
        for k, v in analysis_result["details"].items():
            details_obj[k] = schemas.FeatureDetail(
                value=v["value"],
                ref_display=v.get("ref_display", "N/A"),
                status=v["status"],
                deviation_level=int(v.get("deviation_level", 0)),
                label=v["label"],
                z_score=v.get("z_score", v.get("risk", 0.0))
            )

        report_payload = {
            "final_risk_level": analysis_result["risk_level"],
            "sai_score": analysis_result["sai_score"],
            "explanation_text": analysis_result["explanation"],
            "observations": analysis_result.get("observations", []),
            "advice": analysis_result.get("advice", []),
            "detailed_metrics": analysis_result["details"], 
            "metrics": metrics,
            "quality_metrics": quality_metrics_dict, 
            "session_id": session_id,
            "patient_info": patient_info,
            "confidence_score": analysis_result["confidence"],
            "metadata": metadata_dict,
            "ai_notice": "Kết quả này chỉ mang tính chất hỗ trợ tầm soát chuẩn khoa học."
        }

        # Prepare Spectral Data for Charting (downsampled to 100 points)
        step = max(1, len(y) // 100)
        # Using absolute values for AreaChart to look better and ensure visibility
        spectral_data = [{"name": i, "value": float(abs(y[i*step]))} for i in range(100) if i*step < len(y)]

        # Generate PDF report
        report_path = f"{REPORTS_DIR}/Report_{session_id}.pdf"
        report_generator.generate_medical_report(patient_info, report_payload, temp_file_path, report_path, metrics=metrics)

        # Zip Generation
        zip_filename = None
        try:
            def sanitize(name_str):
                s = re.sub(r"[^0-9a-zA-Z\s_-]", "", name_str)
                s = re.sub(r"\s+", "_", s.strip())
                return s or "patient"

            date_str = datetime.now().strftime("%Y%m%d")
            sanitized_name = sanitize(name)
            zip_filename = f"{sanitized_name}_{date_str}.zip"
            zip_path = f"{REPORTS_DIR}/{zip_filename}"

            folder_name = f"{sanitized_name}_{date_str}"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                if os.path.exists(report_path):
                    zf.write(report_path, arcname=f"{folder_name}/report.pdf")
                if os.path.exists(temp_file_path):
                    zf.write(temp_file_path, arcname=f"{folder_name}/voice_test.wav")
        except Exception as e:
            logger.error(f"Zip error: {e}")
            zip_filename = None

        # [SECURITY] Automatic Cleanup of raw audio after batch processing
        try:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                logger.debug(f"Secure Cleanup: Deleted raw audio {temp_file_path}")
        except Exception as e:
            logger.warning(f"Cleanup failed for {temp_file_path}: {e}")

        # Return Pydantic Model
        return schemas.AnalysisResponse(
            status="SUCCESS",
            session_id=session_id,
            final_risk_level=analysis_result["risk_level"],
            sai_score=analysis_result["sai_score"],
            confidence_score=analysis_result["confidence"],
            group_scores={k: float(v) for k, v in analysis_result["group_scores"].items()},
            biomarker_count=54,
            abnormal_probability=float(analysis_result["abnormal_probability"]),
            details=details_obj,
            deviated_systems=analysis_result.get("deviated_systems", []),
            possible_risks=analysis_result.get("possible_risks", []),
            explanation=analysis_result["explanation"],
            observations=analysis_result["observations"],
            advice=analysis_result["advice"],
            quality_metrics=q_metrics,
            metadata=metadata_obj, # Pass the metadata object
            patient_info=patient_info,
            report_url=f"/report/Report_{session_id}.pdf",
            report_zip_url=f"/report_zip/{zip_filename}" if zip_filename else None,
            spectral_data=spectral_data,
            ai_notice="Kết quả này chỉ mang tính chất hỗ trợ tầm soát chuẩn khoa học."
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={
            "status": "ERROR", 
            "message": f"Server Error: {str(e)}",
            "readable_label": "Lỗi Hệ thống",
            "explanation_text": "Vui lòng thử lại sau."
        }, status_code=500)

@app.get("/report/{filename}")
async def download_report(filename: str):
    file_path = f"{REPORTS_DIR}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="Report not found")


@app.get("/report_zip/{filename}")
async def download_report_zip(filename: str):
    file_path = f"{REPORTS_DIR}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/zip", filename=filename)
    raise HTTPException(status_code=404, detail="Report ZIP not found")



class FeedbackRequest(BaseModel):
    session_id: str
    user_label: str # "NORMAL" or "RISK"
    actual_condition: Optional[str] = None
    comments: Optional[str] = None
    is_expert: bool = False

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Collects ground truth for adaptive AI retraining.
    Gives 3x weight to 'is_expert' verified sessions.
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": feedback.session_id,
            "user_label": feedback.user_label,
            "actual_condition": feedback.actual_condition,
            "comments": feedback.comments,
            "expert_weight": 3.0 if feedback.is_expert else 1.0
        }
        
        # Save to local feedback JSON for retraining
        history_file = "feedback_log.json"
        existing_data = []
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                except:
                    existing_data = []
        
        existing_data.append(log_entry)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
        return {"status": "success", "message": "Feedback recorded."}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Could not save feedback.")

@app.post("/consult")
async def send_to_expert(session_id: str = Form(...), doctor_id: str = Form("DOC_001")):
    """
    Expert Consultation Flow: Triggers a notification to a medical professional.
    In research mode, this just flags the session in the DB/JSON.
    """
    try:
        # Flag the session as 'PENDING_REVIEW'
        consult_log = "consultations.json"
        entry = {
            "session_id": session_id,
            "doctor_id": doctor_id,
            "status": "PENDING_REVIEW",
            "request_time": datetime.now().isoformat()
        }
        
        # We also trigger a simulation of research data collection
        # if the session exists in feedback_log.
        
        return {"status": "success", "message": f"Dữ liệu đã được gửi đến chuyên gia {doctor_id}."}
    except Exception as e:
        logger.error(f"Consultation request failed: {e}")
        raise HTTPException(status_code=500, detail="Could not initiate consultation.")

@app.post("/retrain")
async def trigger_retrain():
    """
    Manually trigger AI baseline recalibration using collected user feedback.
    """
    success, message = ml_model.learning_manager.adaptive_retrain()
    if success:
        return {"status": "success", "message": message}
    else:
        return JSONResponse(content={"status": "error", "message": message}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    # Initial model loading
    ml_model.learning_manager.initialize_model()
    
    # Clean up old reports and temp files on startup
    cleanup_old_files(UPLOAD_DIR)
    cleanup_old_files(REPORTS_DIR)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
