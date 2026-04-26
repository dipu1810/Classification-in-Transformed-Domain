import os
import glob
import numpy as np
import pandas as pd
import cv2
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# IMPORT THE LIBRARY
from mrmr import mrmr_classif

# --- CONFIGURATION ---
IMG_SIZE = 224
DATASET_PATH = r"C:\Users\Mayukh\Documents\KGP\Sem8\Transformed Domain\Breast Cancer\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria\200X"

# -----------------------------------------------------------
# 1. Data Loading
# -----------------------------------------------------------
def get_details_from_filename(filename):
    match = re.search(r"SOB_([BM])_([A-Z]+)-(\d+-[A-Za-z0-9]+)", filename)
    if match: return match.group(1), match.group(3) 
    return None, None

def load_data_grouped(data_path, img_size):
    image_paths = sorted(glob.glob(os.path.join(data_path, '**', '*.png'), recursive=True))
    print(f"Found {len(image_paths)} images.")
    label_map = {'B': 0, 'M': 1}
    
    X_images, y_labels, groups_pid = [], [], []

    for path in image_paths:
        filename = os.path.basename(path)
        binary_char, pid = get_details_from_filename(filename)
        if binary_char is None or pid is None: continue

        img = cv2.imread(path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))

        X_images.append(img)
        y_labels.append(label_map[binary_char])
        groups_pid.append(pid)

    return np.array(X_images), np.array(y_labels), np.array(groups_pid)

# -----------------------------------------------------------
# 2. Feature Extraction (YCrCb Concatenated)
# -----------------------------------------------------------
def extract_concatenated_ycrcb_features(images):
    print(f"   Processing {len(images)} images: YCrCb -> FFT -> Concatenating...")
    n_samples = len(images)
    n_total_features = (IMG_SIZE * IMG_SIZE) * 3
    X_fused = np.zeros((n_samples, n_total_features))

    for i in range(n_samples):
        ycrcb = cv2.cvtColor(images[i], cv2.COLOR_RGB2YCrCb)
        
        # Normalize & FFT
        y_fft = np.abs(np.fft.fft2(ycrcb[:,:,0].astype('float') / 255.0)).flatten()
        cr_fft = np.abs(np.fft.fft2(ycrcb[:,:,1].astype('float') / 255.0)).flatten()
        cb_fft = np.abs(np.fft.fft2(ycrcb[:,:,2].astype('float') / 255.0)).flatten()
        
        X_fused[i] = np.concatenate([y_fft, cr_fft, cb_fft])

    return X_fused

# -----------------------------------------------------------
# 3. Simplified Optimization with mrmr Library
# -----------------------------------------------------------
def optimize_with_mrmr_lib(X, y, max_n=30, pre_filter_size=500):
    print(f"   [Feature Selection] Starting on {X.shape[1]} features...")
    
    # --- STEP 1: Fast Pre-filter (ANOVA) ---
    # We MUST do this because mrmr library is too slow for 150k features
    print(f"     > Pre-filtering top {pre_filter_size} features via ANOVA...")
    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores)
    
    # Get indices of top 500
    top_indices = np.argsort(f_scores)[-pre_filter_size:][::-1]
    X_subset = X[:, top_indices]
    
    # Convert to DataFrame for mrmr library (it prefers named columns)
    # We name them simply '0', '1', '2'... corresponding to their index in X_subset
    X_df = pd.DataFrame(X_subset, columns=[str(i) for i in range(X_subset.shape[1])])
    y_series = pd.Series(y)
    
    # --- STEP 2: mRMR Library Ranking ---
    print(f"     > Running mrmr_classif library to rank top {max_n} features...")
    
    
    selected_cols = mrmr_classif(X=X_df, y=y_series, K=max_n)
    
    # Convert column names back to integer indices relative to X_subset
    selected_indices_local = [int(col) for col in selected_cols]
    
    # Map back to GLOBAL indices (in the original 150k vector)
    global_ranked_indices = top_indices[selected_indices_local]

    # --- STEP 3: Wrapper Optimization (CV) ---
    print(f"     > Wrapper Optimization: Sweeping n=2 to {max_n}...")
    acc_scores = []
    n_range = range(2, max_n + 1)
    
    for n in n_range:
        current_indices = global_ranked_indices[:n]
        current_feats = X[:, current_indices]
        
        # 3-Fold CV
        clf_wrapper = SVC(kernel='linear')
        scores = cross_val_score(clf_wrapper, current_feats, y, cv=3, scoring='accuracy')
        acc_scores.append(scores.mean())
        
    best_n_idx = np.argmax(acc_scores)
    best_n = n_range[best_n_idx]
    best_acc = acc_scores[best_n_idx]
    
    print(f"     > Best n selected: {best_n} (CV Accuracy: {best_acc:.2%})")
    
    return global_ranked_indices[:best_n]

# -----------------------------------------------------------
# 4. Patient Metric Calculation
# -----------------------------------------------------------
def calculate_patient_recognition_rate(y_true, y_pred, p_ids):
    unique_patients = np.unique(p_ids)
    patient_scores = []
    
    for pid in unique_patients:
        indices = np.where(p_ids == pid)[0]
        n_rec = np.sum(y_true[indices] == y_pred[indices])
        patient_scores.append(n_rec / len(indices))
        
    return np.mean(patient_scores)

# -----------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------
if __name__ == "__main__":
    try:
        X_all, y_all, groups_all = load_data_grouped(DATASET_PATH, IMG_SIZE)
    except Exception as e:
        print(f"Error: {e}")
        exit()
        
    if len(X_all) == 0:
        print("No images found. Check path.")
        exit()

    # Pre-extract massive feature matrix
    X_fused_all = extract_concatenated_ycrcb_features(X_all)

    # Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=4)
    fold_patient_accuracies = []

    print("\nStarting Pipeline (mRMR Library + Wrapper CV)...")

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_fused_all, y_all, groups_all)):
        print(f"\n================ FOLD {fold + 1} ===================")
        
        X_train = X_fused_all[train_idx]
        y_train = y_all[train_idx]
        X_test = X_fused_all[val_idx]
        y_test = y_all[val_idx]
        groups_test = groups_all[val_idx] 
        
        print("   Standardizing...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Simplified Selection using mRMR library
        best_features_indices = optimize_with_mrmr_lib(X_train, y_train, max_n=30)
        
        X_train_final = X_train[:, best_features_indices]
        X_test_final = X_test[:, best_features_indices]
        
        print("   Training Final Random Forest...")
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42)
        clf.fit(X_train_final, y_train)
        
        y_pred = clf.predict(X_test_final)
        
        patient_acc = calculate_patient_recognition_rate(y_test, y_pred, groups_test)
        fold_patient_accuracies.append(patient_acc)
        
        print(f"   > Patient-Level Recognition Rate: {patient_acc*100:.2f}%")
        print("   > Confusion Matrix (Image Level):")
        print(confusion_matrix(y_test, y_pred))

    print(f"\nFINAL AVERAGE PATIENT RECOGNITION RATE: {np.mean(fold_patient_accuracies)*100:.2f}%")