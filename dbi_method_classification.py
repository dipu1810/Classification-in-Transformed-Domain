import os
import glob
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, davies_bouldin_score

# --- CONFIGURATION ---
IMG_SIZE = 224
DATASET_PATH = r"C:\Users\Mayukh\Documents\KGP\Sem8\Transformed Domain\Breast Cancer\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria\400X"

# -----------------------------------------------------------
# 1. Data Loading (Grouped)
# -----------------------------------------------------------

def optimize_features_dbi_pre_sorted(X_raw, y, fold_num, max_n=100):
    """
    Assumes X_raw is ALREADY sorted row-wise (e.g., in descending order).
    1. Iterates n from 2 to max_n, selecting the first n columns.
    2. Scales the subset -> Calculates DBI.
    3. Plots DBI vs Number of Top Values Selected.
    """
    print(f"     [Feature Extraction] Starting Pre-sorted DBI analysis on {X_raw.shape[1]} features...")
    
    # Limit max_n so we don't exceed the actual number of features
    actual_max_n = min(max_n, X_raw.shape[1])
    
    # --- Sweep n=2 to max_n and Calculate DBI ---
    dbi_scores = []
    n_range = range(2, actual_max_n + 1)
    
    for n in n_range:
        # Take the top n values directly since X_raw is already sorted
        current_subset_raw = X_raw[:, :n]
        
        # Scale ONLY this subset for the DBI calculation
        current_subset_scaled = StandardScaler().fit_transform(current_subset_raw)
        
        # Calculate DBI using ground truth labels (Internal Class Separability)
        score = davies_bouldin_score(current_subset_scaled, y)
        dbi_scores.append(score)
        
    # --- Find Best n (Minimum DBI is best) ---
    best_idx_in_list = np.argmin(dbi_scores)
    best_n = n_range[best_idx_in_list]
    best_dbi = dbi_scores[best_idx_in_list]
    
    print(f"     > Best n selected: {best_n} (DBI: {best_dbi:.4f})")
    
    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(n_range, dbi_scores, marker='o', linestyle='-', color='g')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'Best n={best_n}')
    plt.title(f'Fold {fold_num}: DBI vs Number of Top Values per Row (Pre-sorted)')
    plt.xlabel('Number of Top Values Selected')
    plt.ylabel('Davies-Bouldin Index')
    plt.legend()
    plt.grid(True)
    plt.show() 
    
    # Return just the optimal integer n
    return best_n


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
def normalize_global(img):
    # Convert to float to avoid overflow/truncation
    img = img.astype('float32')
    img_min = np.min(img)
    img_max = np.max(img)
    
    return (img - img_min) / (img_max - img_min)
    
def extract_concatenated_ycrcb_features(images):
    print(f"   Processing {len(images)} images: YCrCb -> FFT -> Concatenating...")
    n_samples = len(images)
    n_total_features = (IMG_SIZE * IMG_SIZE) * 3
    X_fused = np.zeros((n_samples, n_total_features))

    for i in range(n_samples):
        ycrcb = cv2.cvtColor(images[i], cv2.COLOR_RGB2YCrCb)
        y_fft = (np.abs(np.fft.fft2(ycrcb[:,:,0].astype('float') / 255.0))**2).flatten()
        cr_fft = (np.abs(np.fft.fft2(ycrcb[:,:,1].astype('float') / 255.0))**2).flatten()
        cb_fft = (np.abs(np.fft.fft2(ycrcb[:,:,2].astype('float') / 255.0))**2).flatten()

        X_fused[i] = np.concatenate([y_fft, cr_fft, cb_fft])

    return X_fused

# -----------------------------------------------------------
# 3. DBI Optimization (Magnitude Based)
# -----------------------------------------------------------
def optimize_features_dbi_magnitude(X_raw, y, fold_num, max_n=100):
    """
    1. Rank features by Average Magnitude (Descending).
    2. Iterate n from 2 to max_n.
    3. Scale the subset -> Calculate DBI.
    4. Plot DBI vs Number of Features.
    """
    print(f"     [Feature Selection] Starting Magnitude/DBI analysis on {X_raw.shape[1]} features...")
    
    
    feature_magnitudes = np.mean(X_raw, axis=0)
    
    top_indices = np.argsort(feature_magnitudes)[::-1]
    print(top_indices)
    # Limit to max_n for the loop
    top_candidate_indices = top_indices[:max_n]
    
    # --- STEP 2: Sweep n=2 to 100 and Calculate DBI ---
    dbi_scores = []
    n_range = range(2, max_n + 1)
    print(X_raw.shape)
    for n in n_range:
        # Take raw subset
        current_subset_raw = X_raw[:, top_candidate_indices[:n]]
        # Scale ONLY this subset for the DBI calculation
        # (DBI is distance-based, so scaling is still needed for the metric to be valid)
        current_subset_scaled = StandardScaler().fit_transform(current_subset_raw)
        
        # Calculate DBI using ground truth labels (Internal Class Separability)
        score = davies_bouldin_score(current_subset_scaled, y)
        dbi_scores.append(score)
        
    # --- STEP 3: Find Best n (Minimum DBI is best) ---
    best_idx_in_list = np.argmin(dbi_scores)
    best_n = n_range[best_idx_in_list]
    best_dbi = dbi_scores[best_idx_in_list]
    
    print(f"     > Best n selected: {best_n} (DBI: {best_dbi:.4f})")
    
    # --- STEP 4: Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(n_range, dbi_scores, marker='o', linestyle='-', color='g')
    plt.axvline(x=best_n, color='r', linestyle='--', label=f'Best n={best_n}')
    plt.title(f'Fold {fold_num}: DBI vs Number of Features (Ranked by Magnitude)')
    plt.xlabel('Number of Features (Sorted by Avg Magnitude)')
    plt.ylabel('Davies-Bouldin Index')
    plt.legend()
    plt.grid(True)
    plt.show() 
    
    return top_indices[:best_n+3]

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
    sgkf = StratifiedGroupKFold(n_splits=5)
    fold_patient_accuracies = []

    print("\nStarting Pipeline (Magnitude Sort -> DBI Optimization)...")
    f1_positive_scores = []
    f1_negative_scores = []
    accuracy_scores = []
    precision_scores = []
    balanced_accuracy_scores = []
    recall_scores = []
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_fused_all, y_all, groups_all)):
        print(f"\n================ FOLD {fold + 1} ===================")
        
        # Split Data (Keep it RAW here)
        X_train_raw = X_fused_all[train_idx]
        y_train = y_all[train_idx]
        X_test_raw = X_fused_all[val_idx]
        y_test = y_all[val_idx]
        groups_test = groups_all[val_idx] 
        
        # --- FEATURE SELECTION (Using RAW data for Magnitude ranking) ---
        #best_features_indices = optimize_features_dbi_magnitude(X_train_raw, y_train, fold_num=fold+1, max_n=100)
        
        # Select Features

        #Sort X_train and X_test in descending order of magnitude
        X_train_sort=np.sort(X_train_raw, axis=1)[:, ::-1]
        X_test_sort=np.sort(X_test_raw, axis=1)[:, ::-1]

        best_n=optimize_features_dbi_pre_sorted(X_train_sort, y_train, fold_num=fold+1, max_n=100)
        X_train_selected = X_train_sort[:, :best_n]
        X_test_selected = X_test_sort[:, :best_n]

        # --- NOW Standardize ---
        # (Standardize only after selection, before training)
        print("   Standardizing selected features...")
        scaler = StandardScaler()
        X_train_final = scaler.fit_transform(X_train_selected)
        X_test_final = scaler.transform(X_test_selected)
        
        print("   Training Final Random Forest...")
        clf = RandomForestClassifier(n_estimators=20, class_weight='balanced', random_state=42)
        clf.fit(X_train_final, y_train)
        
        y_pred = clf.predict(X_test_final)
        
        patient_acc = calculate_patient_recognition_rate(y_test, y_pred, groups_test)
        fold_patient_accuracies.append(patient_acc)
        
        print(f"   > Patient-Level Recognition Rate: {patient_acc*100:.2f}%")
        print("   > Confusion Matrix (Image Level):")
        print(confusion_matrix(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        tp, fn, fp, tn = cm.ravel()
        accuracy=(tp + tn) / np.sum(cm)
        accuracy_scores.append(accuracy)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0    
        balanced_accuracy= (tp / (tp + fn) + tn / (tn + fp)) / 2 if ((tp + fn) > 0 and (tn + fp) > 0) else 0
        balanced_accuracy_scores.append(balanced_accuracy) 
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_negative= 2 * (tn / (tn + fn) * tn / (tn + fp)) / ((tn / (tn + fn)) + (tn / (tn + fp))) if ((tn / (tn + fn)) + (tn / (tn + fp))) > 0 else 0
        f1_negative_scores.append(f1_negative)
        print(f"F1 Score: {f1:.4f}")
        f1_positive_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    print(f"\nFINAL AVERAGE PATIENT RECOGNITION RATE: {np.mean(fold_patient_accuracies)*100:.2f}%")
    print(f"FINAL AVERAGE ACCURACY: {np.mean(accuracy_scores)*100:.2f}%")
    print(f"FINAL AVERAGE PRECISION: {np.mean(precision_scores)*100:.2f}%")
    print(f"FINAL AVERAGE RECALL: {np.mean(recall_scores)*100:.2f}%")
    print(f"FINAL AVERAGE BALANCED ACCURACY: {np.mean(balanced_accuracy_scores)*100:.2f}%")
    print(f"FINAL AVERAGE F1 SCORE (Positive): {np.mean(f1_positive_scores)*100:.2f}%")
    print(f"FINAL AVERAGE F1 SCORE (Negative): {np.mean(f1_negative_scores)*100:.2f}%")