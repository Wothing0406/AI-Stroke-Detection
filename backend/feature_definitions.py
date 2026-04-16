# Feature Definitions
# Single source of truth for Feature Extraction and Model Training

FEATURE_DIM = 54  # Expanded from 42 to 54 with clinical features

FEATURE_NAMES = [
    # 0-12: MFCC Means
    "mfcc_mean_1", "mfcc_mean_2", "mfcc_mean_3", "mfcc_mean_4", "mfcc_mean_5", 
    "mfcc_mean_6", "mfcc_mean_7", "mfcc_mean_8", "mfcc_mean_9", "mfcc_mean_10", 
    "mfcc_mean_11", "mfcc_mean_12", "mfcc_mean_13",
    
    # 13-25: MFCC Stds
    "mfcc_std_1", "mfcc_std_2", "mfcc_std_3", "mfcc_std_4", "mfcc_std_5", 
    "mfcc_std_6", "mfcc_std_7", "mfcc_std_8", "mfcc_std_9", "mfcc_std_10", 
    "mfcc_std_11", "mfcc_std_12", "mfcc_std_13",
    
    # 26-27: Spectral Centroid
    "spec_cen_mean", "spec_cen_var",
    
    # 28: Zero Crossing Rate
    "zcr_mean",
    
    # 29-30: RMS Energy
    "rms_mean", "rms_var",
    
    # 31-33: Pitch (F0)
    "f0_mean", "f0_std", "f0_range",
    
    # 34-35: Voice Quality
    "jitter", "shimmer",
    
    # 36-39: Rhythm & Pause
    "syllable_rate", "voice_ratio", "pause_count", "mean_pause_duration",
    
    # 40: Tremor
    "tremor",
    
    # 41: HNR
    "hnr",
    
    # === NEW CLINICAL FEATURES (42-55) ===
    
    # 42: Voice Onset Time
    "vot",
    
    # 43: Articulation Rate
    "articulation_rate",
    
    # 44-48: Advanced Pause Patterns
    "pause_frequency",
    "max_pause_duration",
    "pause_variance",
    
    # 49-55: Formant Stability
    "f1_stability",
    "f2_stability",
    "f3_stability",
    "avg_formant_stability",
    "f1_mean",
    "f2_mean",
    "f3_mean"
]

assert len(FEATURE_NAMES) == FEATURE_DIM, f"Expected {FEATURE_DIM} features, got {len(FEATURE_NAMES)}"
