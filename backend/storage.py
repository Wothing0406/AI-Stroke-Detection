import json
import os
import uuid
from datetime import datetime
import numpy as np

# Storage Paths
DATA_DIR = "data"
PROFILES_FILE = os.path.join(DATA_DIR, "user_profiles.json")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SESSIONS_DIR, exist_ok=True)

def _load_profiles():
    if not os.path.exists(PROFILES_FILE):
        return {}
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def _save_profiles(profiles):
    with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

def get_profile(user_id):
    """
    Retrieve user profile and baseline stats.
    """
    profiles = _load_profiles()
    return profiles.get(user_id)

def create_or_update_profile(user_id, name, dob):
    """
    Create a new profile or update existing basic info.
    """
    profiles = _load_profiles()
    
    if user_id not in profiles:
        profiles[user_id] = {
            "id": user_id,
            "name": name,
            "dob": dob,
            "created_at": datetime.now().isoformat(),
            "baseline_metrics": {},
            "session_count": 0
        }
    else:
        # Update info if provided
        if name: profiles[user_id]["name"] = name
        if dob: profiles[user_id]["dob"] = dob
        
    _save_profiles(profiles)
    return profiles[user_id]

def update_baseline(user_id, new_metrics):
    """
    Update the user's running average (baseline) for key metrics.
    Simple Moving Average (SMA) approach for stability.
    """
    profiles = _load_profiles()
    user = profiles.get(user_id)
    
    if not user:
        return None

    current_baseline = user.get("baseline_metrics", {})
    count = user.get("session_count", 0)
    
    # Key metrics to track (Matching 54-biomarker set)
    keys = ["jitter_local", "shimmer_local", "hnr", "mean_f0", "speech_rate", "rms_energy"]
    
    updated_baseline = {}
    
    for k in keys:
        if k in new_metrics:
            val = new_metrics[k]
            if isinstance(val, (list, np.ndarray)):
                val = float(np.mean(val))
            
            old_val = current_baseline.get(k, val)
            
            # Weighted update: 80% history, 20% new (Smoother for clinical stability)
            if count > 0:
                new_avg = (old_val * 0.8) + (val * 0.2)
            else:
                new_avg = val
                
            updated_baseline[k] = new_avg
            
    user["baseline_metrics"] = updated_baseline
    user["session_count"] = count + 1
    user["last_updated"] = datetime.now().isoformat()
    
    _save_profiles(profiles)
    return updated_baseline

def save_session(user_id, session_data):
    """
    Save the full details of a screening session.
    """
    session_id = str(uuid.uuid4())
    filename = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(SESSIONS_DIR, filename)
    
    data = {
        "session_id": session_id,
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        **session_data
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return session_id

def get_comparison(user_id, current_metrics):
    """
    Compare current metrics against user's baseline.
    Returns deviation percentages.
    """
    profile = get_profile(user_id)
    if not profile or not profile.get("baseline_metrics") or profile.get("session_count", 0) == 0:
        return None
    
    baseline = profile["baseline_metrics"]
    comparison = {}
    
    for k, base_val in baseline.items():
        if k in current_metrics:
            curr_val = current_metrics[k]
            if base_val == 0: continue
            
            # Calculate % deviation
            deviation = ((curr_val - base_val) / (base_val + 1e-9)) * 100
            comparison[k] = {
                "baseline": float(base_val),
                "current": float(curr_val),
                "deviation_percent": float(deviation),
                "is_worse": False 
            }
            
            # Determine "Worse" direction
            # Jitter/Shimmer: Higher is worse
            if k in ["jitter_local", "shimmer_local"] and deviation > 15: 
                comparison[k]["is_worse"] = True
                
            # HNR: Lower is worse
            if k == "hnr" and deviation < -15:
                comparison[k]["is_worse"] = True
            
            # Speech Rate: significant change is notable
            if k == "speech_rate" and abs(deviation) > 20: 
                comparison[k]["is_worse"] = True
                
    return comparison
