from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class QualityMetrics(BaseModel):
    snr: float
    vad_ratio: float
    clipping_ratio: float
    pitch_confidence: float
    duration: float
    tier: str

class FeatureDetail(BaseModel):
    value: float
    ref_display: str
    status: str
    deviation_level: int
    label: str
    z_score: Optional[float] = 0.0

class AnalysisMetadata(BaseModel):
    model_version: str = "1.0.0"
    feature_set_version: str = "v2_clinical"
    normalization_method: str = "min_max_clinical"
    threshold_policy: str = "strict_consensus"
    dialect: str = "North" # Added for regional calibration
    analysis_id: str
    timestamp: str

class AnalysisResponse(BaseModel):
    status: str
    session_id: str
    final_risk_level: str
    sai_score: float = 0.0 # Speech Abnormality Index (0-100)
    confidence_score: float
    group_scores: Dict[str, float] # Pitch, Amplitude, Harmonic, etc.
    biomarker_count: int = 54
    abnormal_probability: float
    details: Dict[str, FeatureDetail] # Full 54-biomarker breakdown
    deviated_systems: List[str] = []
    possible_risks: List[str] = []
    explanation: str
    observations: List[str] = []
    advice: List[str] = []
    quality_metrics: QualityMetrics
    metadata: AnalysisMetadata
    patient_info: Dict[str, Any] # Added to restore UI screening info
    report_url: str
    report_zip_url: Optional[str] = None
    spectral_data: List[Dict[str, Any]] = [] # Spectral waveform for charting
    ai_notice: str
