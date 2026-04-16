"""
Reference Standards for Voice Analysis
Defines checking rules and thresholds for Stroke Detection Screening.
Age groups:
- Young: < 40
- Middle: 40-60
- Senior: > 60

Thresholds derived from standard normative acoustic data.
"""

def get_reference_standards(age: int):
    # Default to Middle-aged if age is invalid
    if not isinstance(age, int) or age < 0:
        age = 50

    # 1. Jitter (Frequency Perturbation) - Norm: < 1.04%
    # Seniors allow slightly higher jitter due to natural aging.
    if age < 40:
        jitter_limit = 0.0104 # 1.04%
    elif age <= 60:
        jitter_limit = 0.012  # 1.2%
    else:
        jitter_limit = 0.015  # 1.5%

    # 2. Shimmer (Amplitude Perturbation) - Norm: < 3.81%
    if age < 40:
        shimmer_limit = 0.0381
    elif age <= 60:
        shimmer_limit = 0.045
    else:
        shimmer_limit = 0.055

    # 3. HNR (Harmonic-to-Noise Ratio) - Norm: > 20dB (lower is bad)
    # Stroke/Dysarthria -> breathy voice -> Lower HNR
    if age < 60:
        hnr_limit = 20.0 
    else:
        hnr_limit = 18.0

    # 4. VOT (Voice Onset Time) - Norm: 0.015-0.090 seconds (15-90ms)
    # clinical_features.py outputs VOT in seconds, so we use seconds here.
    vot_min, vot_max = 0.015, 0.090

    # 5. Formant Stability - Norm: > 0.75 stability (0-1 scale)
    if age < 60:
        stability_limit = 0.70  # 70%
    else:
        stability_limit = 0.60  # 60% for seniors

    # 6. Syllable Rate (Articulation) - Norm: 3-6 syllables/sec
    # Slower in stroke (Dysarthria/Apraxia)
    rate_min = 3.0
    
    # 7. Tremor / Run Index
    tremor_limit = 0.05

    return {
        "jitter": {"max": jitter_limit, "unit": "%", "label": "Jitter"},
        "shimmer": {"max": shimmer_limit, "unit": "%", "label": "Shimmer"},
        "hnr": {"min": hnr_limit, "unit": "dB", "label": "HNR"},
        "vot": {"min": vot_min, "max": vot_max, "unit": "ms", "label": "VOT"},
        "f1_stability": {"min": stability_limit, "unit": "idx", "label": "Formant Stability"},
        "articulation_rate": {"min": rate_min, "unit": "syl/s", "label": "Speech Rate"},
        "tremor": {"max": tremor_limit, "unit": "idx", "label": "Tremor Index"}
    }
