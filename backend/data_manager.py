"""
Data Manager for Supervised Learning
Handles loading, balancing, and augmenting labeled voice datasets.
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import audio_processing
import feature_extraction
from feature_definitions import FEATURE_DIM

DATASET_DIR = "dataset"
NORMAL_DIR = os.path.join(DATASET_DIR, "normal")
STROKE_DIR = os.path.join(DATASET_DIR, "stroke")

def load_features_from_folder(folder_path, label):
    """
    Load all audio files from a folder and extract features.
    
    Args:
        folder_path: Path to folder containing .wav files
        label: 0 for normal, 1 for stroke
        
    Returns:
        features: np.array of shape (n_samples, FEATURE_DIM)
        labels: np.array of shape (n_samples,)
    """
    features = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist")
        return np.array([]), np.array([])
    
    print(f"Loading from {folder_path}...")
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            try:
                filepath = os.path.join(folder_path, filename)
                y, sr = audio_processing.load_audio(filepath)
                
                if y is not None:
                    # Validate audio quality
                    is_valid, reasons = audio_processing.validate_audio(y, sr)
                    if not is_valid:
                        print(f"Skip {filename}: {reasons}")
                        continue
                    
                    # Pad/trim to standard duration
                    y = audio_processing.pad_or_trim_audio(
                        y, 
                        duration=audio_processing.DURATION, 
                        sr=sr
                    )
                    
                    # Extract features
                    feat, _ = feature_extraction.extract_features(y, sr)
                    
                    if feat is not None and len(feat) == FEATURE_DIM:
                        features.append(feat)
                        labels.append(label)
                    else:
                        print(f"Skip {filename}: Feature dimension mismatch")
                        
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
    
    print(f"Loaded {len(features)} samples from {folder_path}")
    return np.array(features), np.array(labels)

def load_labeled_data():
    """
    Load all labeled data from dataset folders.
    
    Returns:
        X: Features array (n_samples, FEATURE_DIM)
        y: Labels array (n_samples,) - 0=normal, 1=stroke
    """
    # Load normal samples (label=0)
    X_normal, y_normal = load_features_from_folder(NORMAL_DIR, label=0)
    
    # Load stroke samples (label=1)
    X_stroke, y_stroke = load_features_from_folder(STROKE_DIR, label=1)
    
    # Combine
    if len(X_normal) == 0 and len(X_stroke) == 0:
        print("Warning: No labeled data found!")
        return np.array([]), np.array([])
    
    if len(X_normal) == 0:
        return X_stroke, y_stroke
    
    if len(X_stroke) == 0:
        return X_normal, y_normal
    
    X = np.vstack([X_normal, X_stroke])
    y = np.hstack([y_normal, y_stroke])
    
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(X)}")
    print(f"  Normal: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Stroke: {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    
    return X, y

def balance_dataset(X, y, method='smote'):
    """
    Balance imbalanced dataset using SMOTE or class weights.
    
    Args:
        X: Features
        y: Labels
        method: 'smote' or 'weights'
        
    Returns:
        X_balanced, y_balanced (if SMOTE)
        or original X, y with class_weight dict
    """
    if len(X) == 0:
        return X, y, None
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution before balancing: {dict(zip(unique, counts))}")
    
    # If already balanced (within 30%), no need to balance
    if len(unique) == 2:
        ratio = min(counts) / max(counts)
        if ratio > 0.7:
            print("Dataset is already balanced")
            return X, y, None
    
    if method == 'smote' and len(unique) == 2:
        # SMOTE requires at least 2 samples per class
        min_samples = min(counts)
        if min_samples < 2:
            print(f"Warning: Not enough samples for SMOTE (min={min_samples})")
            # Use class weights instead
            class_weight = {0: len(y)/(2*counts[0]), 1: len(y)/(2*counts[1])}
            return X, y, class_weight
        
        try:
            # Use SMOTE to oversample minority class
            k_neighbors = min(5, min_samples - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            unique, counts = np.unique(y_balanced, return_counts=True)
            print(f"Class distribution after SMOTE: {dict(zip(unique, counts))}")
            
            return X_balanced, y_balanced, None
        except Exception as e:
            print(f"SMOTE failed: {e}. Using class weights instead.")
            class_weight = {0: len(y)/(2*counts[0]), 1: len(y)/(2*counts[1])}
            return X, y, class_weight
    
    else:
        # Use class weights for model training
        if len(unique) == 2:
            class_weight = {0: len(y)/(2*counts[0]), 1: len(y)/(2*counts[1])}
        else:
            class_weight = 'balanced'
        print(f"Using class weights: {class_weight}")
        return X, y, class_weight

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if len(X) == 0:
        return None, None, None, None
    
    # Check if we have enough samples for stratification
    unique, counts = np.unique(y, return_counts=True)
    min_samples = min(counts) if len(counts) > 0 else 0
    
    if min_samples < 2:
        print("Warning: Not enough samples for stratified split. Using random split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return train_test_split(
        X, y, 
        test_size=test_size, 
        stratify=y,
        random_state=random_state
    )

def augment_data(X, y, augmentation_factor=2):
    """
    Simple data augmentation by adding Gaussian noise.
    
    Args:
        X: Features
        y: Labels
        augmentation_factor: How many augmented copies per sample
        
    Returns:
        X_augmented, y_augmented
    """
    if len(X) == 0:
        return X, y
    
    X_aug = [X]
    y_aug = [y]
    
    for i in range(augmentation_factor - 1):
        # Add small Gaussian noise (5% of std)
        noise = np.random.normal(0, 0.05 * np.std(X, axis=0), X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)
    
    X_augmented = np.vstack(X_aug)
    y_augmented = np.hstack(y_aug)
    
    print(f"Augmented dataset: {len(X)} -> {len(X_augmented)} samples")
    return X_augmented, y_augmented

if __name__ == "__main__":
    # Test data loading
    print("=== Testing Data Manager ===")
    X, y = load_labeled_data()
    
    if len(X) > 0:
        print(f"\nFeature shape: {X.shape}")
        print(f"Label shape: {y.shape}")
        
        # Test balancing
        X_bal, y_bal, weights = balance_dataset(X, y, method='smote')
        
        # Test train/test split
        X_train, X_test, y_train, y_test = get_train_test_split(X_bal, y_bal)
        if X_train is not None:
            print(f"\nTrain set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
    else:
        print("\nNo data available. Please add audio files to dataset/normal or dataset/stroke")
