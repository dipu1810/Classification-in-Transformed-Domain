import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from mrmr import mrmr_classif
import scipy.fftpack as dct

# ==========================================
# 1. Load & Preprocess Data
# ==========================================
def load_and_preprocess_mnist():
    print("Loading MNIST Dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0
    return (x_train, y_train), (x_test, y_test)

# ==========================================
# 2. Domain Transformation (DCT)
# ==========================================
def apply_dct_and_flatten(images):
    # Applying 2D DCT
    F = dct.dct(dct.dct(images, axis=1, norm='ortho'), axis=2, norm='ortho')
    mag = np.abs(F)
    return mag.reshape(mag.shape[0], -1)

# ==========================================
# 3. Main Experiment Execution
# ==========================================
def run_top_70_evaluation_mlp():
    TARGET_FEATURES = 70
    SUBSAMPLE_SIZE = 5000  # Subsample size for mRMR memory/speed efficiency
    
    # --- Step A: Load and Transform ---
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    
    print("\nApplying DCT transformation...")
    x_train_freq = apply_dct_and_flatten(x_train)
    x_test_freq = apply_dct_and_flatten(x_test)

    # --- Step B: Run mRMR to get exactly top 70 ---
    print(f"\nRunning mRMR to select the top {TARGET_FEATURES} informative frequencies...")
    np.random.seed(42)
    subsample_idx = np.random.choice(len(x_train_freq), SUBSAMPLE_SIZE, replace=False)
    
    X_sub = pd.DataFrame(x_train_freq[subsample_idx])
    y_sub = pd.Series(y_train[subsample_idx])

    # K is set to TARGET_FEATURES (70)
    selected_columns = mrmr_classif(X=X_sub, y=y_sub, K=TARGET_FEATURES)
    ranked_indices = [int(col) for col in selected_columns]
    
    print(f"\nSuccessfully selected {TARGET_FEATURES} features.")
    
    # --- Step C: Train and Evaluate using MLP ---
    print(f"Training Multi-Layer Perceptron (MLP) on the top {TARGET_FEATURES} features...")
    
    # Slice the dataset to only include the 70 selected features
    x_train_selected = x_train_freq[:, ranked_indices]
    x_test_selected = x_test_freq[:, ranked_indices]
    
    # Initialize and train the MLP
    # Using 2 hidden layers (128 and 64 neurons) and early stopping to prevent overfitting
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        max_iter=500, 
        early_stopping=True, 
        random_state=42
    )
    
    mlp.fit(x_train_selected, y_train)
    
    # Predict and calculate accuracy
    preds = mlp.predict(x_test_selected)
    acc = accuracy_score(y_test, preds) * 100
    
    print(f"Final Test Accuracy using Top {TARGET_FEATURES} mRMR Features with MLP: {acc:.2f}%")

if __name__ == "__main__":
    run_top_70_evaluation_mlp()