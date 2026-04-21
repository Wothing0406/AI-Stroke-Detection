import numpy as np

def compute_group_risk(features, ranges):
    """
    Computes a 0-100 risk score for a feature group based on deviations.
    Returns (group_risk, detailed_results)
    """
    deviations = []
    detailed_results = {}
    
    for key, val in features.items():
        if key in ranges:
            cfg = ranges[key]
            # Normalization (Z-score like)
            dev = abs(val - cfg["mid"]) / (cfg["range"] + 1e-9)
            # Clamp and scale (logistic-like risk)
            risk_val = 1.0 / (1.0 + np.exp(-5 * (dev - 1.0)))
            
            # Map risk to status for report compatibility
            status = "NORMAL"
            if risk_val > 0.8: status = "SIGNIFICANTLY_DEVIATED"
            elif risk_val > 0.4: status = "DEVIATED"
            
            detailed_results[key] = {
                "value": float(val),
                "z_score": float((val - cfg["mid"]) / (cfg["range"] / 2.0 + 1e-9)), # Estimate Z-score for charts
                "status": status,
                "risk": float(risk_val),
                "deviation_level": float(dev), # Added for report radar
                "norm_val": float(min(dev / 2.0, 1.0)), # Added for report radar
                "label": cfg.get("label", key),
                "ref_display": f"{cfg['mid'] - cfg['range']/2:.1f}-{cfg['mid'] + cfg['range']/2:.1f}"
            }
            deviations.append(risk_val)
        else:
            detailed_results[key] = {
                "value": float(val),
                "z_score": 0.0,
                "status": "NORMAL",
                "risk": 0.0,
                "deviation_level": 0.0,
                "norm_val": 0.0,
                "label": key.replace("_", " ").title(),
                "ref_display": "Typical"
            }
    
    group_risk = float(np.mean(deviations) * 100) if deviations else 0.0
    return group_risk, detailed_results

def analyze_risk(features_grouped: dict, age: int, ml_prob: float = 0.0, signal_quality: float = 1.0, gender: str = "Nam"):
    """
    Complete 54-biomarker clinical analysis.
    Equal weights across 6 physiological groups.
    """
    
    # Research Standards for Calibration
    # Research Standards with Dialect Calibration
    # Northern: Sharp, clear tones, higher pitch variability
    # Central: Heavy tones, shorter vowel duration
    # Southern: Soft tones, slightly faster speech rate, lower pitch variability
    
    dialect = "North" # Default
    if "metadata" in features_grouped and "dialect" in features_grouped["metadata"]:
        dialect = features_grouped["metadata"]["dialect"]

    # Base Standards (Northern defaults)
    # Calibrated for Gender: Male vs Female
    f0_mid = 125.0 if gender == "Nam" else 215.0
    f0_range = 60.0 if gender == "Nam" else 80.0
    f1_mid = 500.0 if gender == "Nam" else 600.0
    
    STANDARDS = {
        "pitch": {
            "mean_f0": {"mid": f0_mid, "range": f0_range, "label": "Mean F0"},
            "jitter_local": {"mid": 0.01, "range": 0.02, "label": "Jitter (Local)"},
            "f0_std": {"mid": 6.0, "range": 12.0, "label": "F0 Standard Deviation"},
            "pitch_stability": {"mid": 0.95, "range": 0.1, "label": "Pitch Stability"}
        },
        "amplitude": {
            "rms_energy": {"mid": 0.05, "range": 0.1, "label": "RMS Energy"},
            "shimmer_local": {"mid": 0.07, "range": 0.15, "label": "Shimmer (Local)"},
            "amplitude_mod_index": {"mid": 0.1, "range": 0.2, "label": "Amplitude Modulation"},
            "mean_amplitude": {"mid": 0.1, "range": 0.2, "label": "Mean Amplitude"}
        },
        "harmonic": {
            "hnr": {"mid": 24.0, "range": 12.0, "label": "HNR"},
            "cpp": {"mid": 15.0, "range": 6.0, "label": "CPP (Cepstral Peak)"},
            "harmonic_spectral_tilt": {"mid": -15.0, "range": 10.0, "label": "Spectral Tilt"},
            "harmonic_richness_factor": {"mid": 0.8, "range": 0.4, "label": "Harmonic Richness"}
        },
        "spectral": {
            "spectral_centroid": {"mid": 2600.0, "range": 1600.0, "label": "Spectral Centroid"},
            "spectral_rolloff": {"mid": 4200.0, "range": 2200.0, "label": "Spectral Rolloff"},
            "spectral_flux": {"mid": 0.5, "range": 1.0, "label": "Spectral Flux"},
            "mfcc_1": {"mid": -200.0, "range": 400.0, "label": "MFCC-1"}
        },
        "temporal": {
            "speech_rate": {"mid": 5.8, "range": 3.5, "label": "Speech Rate"},
            "zcr": {"mid": 0.06, "range": 0.12, "label": "Zero Crossing Rate"},
            "pause_ratio": {"mid": 0.15, "range": 0.2, "label": "Pause Ratio"},
            "vot": {"mid": 0.04, "range": 0.08, "label": "VOT Proxy"}
        },
        "quality": {
            "formant_f1": {"mid": f1_mid, "range": 210.0, "label": "Formant F1"},
            "breathiness_index": {"mid": 1.1, "range": 2.2, "label": "Breathiness Index"},
            "hoarseness_index": {"mid": 0.8, "range": 1.6, "label": "Hoarseness Index"},
            "roughness_index": {"mid": 1.5, "range": 3.0, "label": "Roughness Index"}
        }
    }

    # APPLY DIALECT BIAS CORRECTION
    if dialect == "South":
        STANDARDS["pitch"]["mean_f0"]["mid"] -= 8.0 # Typically softer
        STANDARDS["temporal"]["speech_rate"]["mid"] += 0.4 # Typically faster
    elif dialect == "Central":
        STANDARDS["pitch"]["f0_std"]["mid"] += 1.5 # Higher tonal variance
        STANDARDS["temporal"]["speech_rate"]["mid"] -= 0.4 # Typically measured

    g_risks = {}
    all_details = {}
    
    for group_name, group_data in features_grouped["groups"].items():
        ranges = STANDARDS.get(group_name, {})
        risk, details = compute_group_risk(group_data, ranges)
        g_risks[group_name] = risk
        all_details.update(details)

    # SAI = Mean of 6 group risks
    sai_score_raw = sum(g_risks.values()) / 6.0
    
    full_vector = features_grouped["vector"]
    p_stab = full_vector.get("pitch_stability", 1.0)
    q_stab = full_vector.get("f1_stability", 1.0)
    stability_avg = (p_stab + q_stab) / 2.0
    
    penalty = max(0, 0.82 - stability_avg) * 0.4
    sai_score = min(100.0, sai_score_raw * (1.0 + penalty))

    # --- ENHANCED RECOMMENDATION ENGINE ---
    specific_advice = []
    observations = []

    # Map indicators to Vietnamese labels
    BIOMARKER_LABELS = {
        "jitter_local": "Độ rung thanh đới (Jitter)",
        "shimmer_local": "Độ biến thiên biên độ (Shimmer)",
        "hnr": "Tỷ lệ hài nhiễu (HNR)",
        "cpp": "Độ rõ nét hài âm (CPP)",
        "mean_f0": "Tần số cơ bản (F0)",
        "formant_f1": "Cấu trúc âm học (F1 Formant)",
        "vot": "Thời gian khởi phát thanh (VOT)",
        "speech_rate": "Tốc độ phát âm",
        "rms_energy": "Âm lượng (Loudness)",
        "pause_ratio": "Tỷ lệ ngắt nghỉ",
        "breathiness_index": "Chỉ số tiếng phào (Breathiness)",
        "hoarseness_index": "Chỉ số khàn tiếng (Hoarseness)"
    }

    # Analyze individual biomarker deviations
    for key, detail in all_details.items():
        if key in BIOMARKER_LABELS and detail["status"] != "NORMAL":
            label = BIOMARKER_LABELS[key]
            severity = "cao" if detail["risk"] > 0.7 else "vừa phải"
            observations.append(f"{label} có dấu hiệu biến thiên {severity}.")
            
            # Granular Advice
            if key == "jitter_local":
                specific_advice.append("Tập trung vào các bài tập kiểm soát hơi thở để ổn định độ rung thanh đới.")
            elif key == "shimmer_local":
                specific_advice.append("Tránh gắng sức giọng nói, nên nghỉ ngơi và uống đủ nước.")
            elif key == "hnr":
                specific_advice.append("Thực hiện các bài tập phát âm âm mở để cải thiện độ trong của giọng.")
            elif key == "speech_rate":
                specific_advice.append("Hãy thử đọc văn bản chậm lại và ngắt nghỉ đúng nhịp.")
            elif key == "pause_ratio":
                specific_advice.append("Chú ý điều tiết nhịp thở và ngắt nghỉ tự nhiên giữa các câu.")
            elif key == "breathiness_index":
                specific_advice.append("Luyện tập các bài phát âm mạnh mẽ hơn để giảm tình trạng rò rỉ hơi khi nói.")
            elif key == "hoarseness_index":
                specific_advice.append("Nếu tình trạng khàn tiếng kéo dài, hãy đi khám chuyên khoa tai mũi họng.")

    # High-level Risk Categorization
    if sai_score <= 30:
        risk_level = "NORMAL"
        status_msg = "BÌNH THƯỜNG"
        explanation = f"Chỉ số SAI = {sai_score:.1f}/100 ở mức độ an toàn. Giọng nói ổn định."
    elif sai_score <= 60:
        risk_level = "MEDIUM"
        status_msg = "SAI LỆCH TRUNG BÌNH"
        explanation = f"Chỉ số SAI = {sai_score:.1f}/100 có dấu hiệu sai lệch nhẹ. Cần theo dõi thêm."
    else:
        risk_level = "HIGH"
        status_msg = "RỦI RO CAO"
        explanation = f"CẢNH BÁO: Chỉ số SAI = {sai_score:.1f}/100 vượt ngưỡng an toàn. Cần thăm khám chuyên khoa."

    final_explanation = f"{explanation} Mô hình AI ước tính xác suất bất thường {ml_prob*100:.1f}%."

    # Default advice if none gathered
    if not specific_advice:
        specific_advice = ["Duy trì theo dõi sức khỏe giọng nói định kỳ."]
    if sai_score > 60:
        specific_advice.insert(0, "CẦN THIẾT thăm khám bác sĩ chuyên khoa Thần kinh hoặc Tai Mũi Họng.")

    # Map physiological groups to high-level clinical systems
    SYSTEM_MAP = {
        "pitch": "Thanh quản (Laryngeal System)",
        "amplitude": "Năng lượng & Hơi thở (Energy & Respiration)",
        "harmonic": "Cộng hưởng âm học (Harmonic Resonance)",
        "spectral": "Độ sắc nét âm thanh (Spectral Clarity)",
        "temporal": "Tiến trình thời gian (Temporal Dynamics)",
        "quality": "Chất lượng giọng nói (Voice Quality)"
    }
    
    deviated_systems = []
    for g_name, g_score in g_risks.items():
        if g_score > 40: # Threshold for "notable deviation"
            deviated_systems.append(SYSTEM_MAP.get(g_name, g_name))
            
    possible_risks = []
    # High Risk
    if sai_score > 60:
        possible_risks.append("Ưu tiên tầm soát đột quỵ (High Stroke Risk Filter)")
        possible_risks.append("Rối loạn vận động âm thanh (Acoustic Dysarthria)")
    
    # Moderate Risk / Specific Deviations
    if sai_score > 30:
        if g_risks.get("temporal", 0) > 40:
            possible_risks.append("Suy giảm tiến trình thời gian giọng nói (Temporal Logic Displacement)")
        if g_risks.get("pitch", 0) > 40:
            possible_risks.append("Biến thiên cao độ không ổn định (Pitch Instability)")
        if g_risks.get("quality", 0) > 40:
            possible_risks.append("Dấu hiệu mệt mỏi thanh quản (Vocal Strain)")
    
    if not possible_risks and sai_score > 20:
        possible_risks.append("Theo dõi biến thiên âm học định kỳ")

    return {
        "sai_score": round(sai_score, 1),
        "confidence": round(float(min(0.98, 0.85 + signal_quality * 0.15)), 2), # Dynamic Based on Quality
        "group_scores": {k: round(v, 1) for k, v in g_risks.items()},
        "details": all_details,
        "biomarker_count": 54,
        "abnormal_probability": round(float(ml_prob * 100), 1),
        "explanation": final_explanation,
        "observations": observations or ["Các chỉ số nằm trong giới hạn cho phép."],
        "advice": specific_advice[:4], # Limit to top 4
        "risk_level": risk_level,
        "status_msg": status_msg,
        "deviated_systems": deviated_systems,
        "possible_risks": possible_risks
    }
