import numpy as np
import joblib
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "model.joblib"

def train_dummy_model():
    """
    Train a dummy Support Vector Classification (SVC) model for demonstration.
    
    REAL DATA INTEGRATION NOTE:
    To use real data:
    1. Collect audio files for 'Healthy' and 'Stroke' classes.
    2. Extract features using feature_extraction.py for all files.
    3. Create X (features array) and y (labels array).
    4. Call fit() on the pipeline with real X and y.
    """
    print("Training dummy model (SVM)...")
    
    # Simulate feature vectors (Length 29: 13 MFCC means + 13 MFCC stds + Centroid + ZCR + RMS)
    n_features = 29
    n_samples = 100
    
    # Class 0: Healthy
    X_healthy = np.random.normal(loc=0.5, scale=0.5, size=(n_samples // 2, n_features))
    y_healthy = np.zeros(n_samples // 2)
    
    # Class 1: At Risk (Simulating higher variance/irregularity)
    X_risk = np.random.normal(loc=2.0, scale=1.0, size=(n_samples // 2, n_features))
    y_risk = np.ones(n_samples // 2)
    
    X = np.vstack([X_healthy, X_risk])
    y = np.hstack([y_healthy, y_risk])
    
    # Create pipeline: Scale features -> SVM
    # Probability=True is needed to get confidence scores
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', probability=True))
    clf.fit(X, y)
    
    # Save using joblib (better for sklearn models than pickle)
    joblib.dump(clf, MODEL_PATH)
        
    print(f"Model saved to {MODEL_PATH}")
    return clf

def load_model():
    """
    Load the trained model from disk.
    """
    if not os.path.exists(MODEL_PATH):
        return train_dummy_model()
        
    return joblib.load(MODEL_PATH)

def predict_risk(features):
    """
    Predict the risk level based on extracted features.
    Returns: (label, confidence_score)
    """
    clf = load_model()
    
    # Reshape features for prediction (1, n_features)
    features = features.reshape(1, -1)
    
    # Predict
    prob = clf.predict_proba(features)[0]
    prediction = clf.predict(features)[0]
    
    risk_label = "High Risk" if prediction == 1 else "Low Risk"
    confidence = prob[int(prediction)]
    
    return risk_label, confidence

if __name__ == "__main__":
    train_dummy_model()
