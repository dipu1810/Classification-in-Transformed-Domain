#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import os
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import time
from scipy.fftpack import dct,idct
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)        
    

#
# Verify Reading Dataset via MnistDataloader class
#

import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#
# Set file paths based on added MNIST Datasets
#
input_path = r'C:\Users\Mayukh\Documents\KGP\Sem8\MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 2
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1

#
# Load MINST dataset
#

def create_masks(x,y,percentage):
    net_mag=np.zeros((10,28,28))
    F=dct(dct(x, axis=1, norm='ortho'), axis=2, norm='ortho')
    for i in range(0, len(x)):
        mag = np.abs(F[i])
        net_mag[y[i]]+=mag

    binary_mask = np.zeros((10, 28, 28), dtype=int)

    for i in range(10):
        N = net_mag[i].size
        K = max(1, int(percentage * N))

        flat = net_mag[i].flatten()
        threshold = np.partition(flat, -K)[-K]

        binary_mask[i] = (net_mag[i] >= threshold).astype(int)
    union_mask = np.any(binary_mask, axis=0)
    return union_mask


def compute_psnr(images,mask):
    F=dct(dct(images, axis=1, norm='ortho'), axis=2, norm='ortho')
    masked_F=F*mask
    images_hat=idct(idct(masked_F, axis=2, norm='ortho'), axis=1, norm='ortho')
    mse=np.mean((images-images_hat)**2,axis=(1,2))
    psnr_list=[]
    for m in mse:
        if(m==0):
            psnr_list.append(float('inf'))
        else:
            psnr_list.append(20*np.log10(1/np.sqrt(m)))
    return np.mean(psnr_list)
def get_fixed_indices(mask):
    # Flatten the mask and find the index (0 to 783) of every kept coefficient
    flat_mask = mask.flatten()
    indices = np.flatnonzero(flat_mask)
    # Convert to a TF constant so it lives on the GPU/Accelerator
    return tf.constant(indices, dtype=tf.int32)

@tf.function
def tf_dct2d_preprocess(images, indices):
    """
    Computes 2D DCT (Type II) and gathers specific coefficients.
    Input: (Batch, 28, 28) - float32
    Output: (Batch, K) - float32 (Real numbers only!)
    """
    temp = tf.matmul(DCT_MAT, images) # Broadcasts DCT_MAT over the batch
    
    # 2. Matrix Multiplication: (C * X) * C^T
    dct_2d = tf.matmul(temp, DCT_MAT_T)
    
    # 3. Flatten and Gather (Same as before)
    batch_size = tf.shape(images)[0]
    dct_flat = tf.reshape(dct_2d, [batch_size, -1])
    
    # 4. Gather specific coefficients
    out = tf.gather(dct_flat, indices, axis=1)
    
    return out
    
    

def masked_dct_features(images,mask):
    F=dct(dct(images, axis=1, norm='ortho'), axis=2, norm='ortho')
    masked_F=F*mask
    features=masked_F[:,mask==1]
    return features

def train_and_test(x_tr, y_tr, x_val, y_val, x_test, y_test,
                   save_path="best_mlp.keras",
                   epochs=20,
                   batch_size=64,
                   verbose=1):
    start_train_time=time.time()
    mean = x_tr.mean(axis=0, keepdims=True)
    std  = x_tr.std(axis=0, keepdims=True) + 1e-8
    x_tr_norm  = (x_tr - mean) / std
    x_val_norm = (x_val - mean) / std
    input_dim = x_tr_norm.shape[1]
    num_classes = int(max(y_tr.max(), y_val.max(), y_test.max()) + 1)

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(300,activation='relu'),
        #layers.Dense(256, activation='relu'),
        layers.Dense(100, activation='relu'),
        #layers.Dense(64,activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- callbacks: save best by validation accuracy; optional early stopping ---
    checkpoint = ModelCheckpoint(
        filepath=save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    earlystop = EarlyStopping(
        monitor='val_accuracy',
        patience=20,
        restore_best_weights=False  # we will load from saved file instead
    )

    # --- train ---
    history = model.fit(
        x_tr_norm, y_tr,
        validation_data=(x_val_norm, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, earlystop],
        verbose=verbose
    )
    end_train_time=time.time()
    x_test_norm= (x_test - mean) / std
    # --- load the saved best model (guaranteed to exist if training ran at least one epoch) ---
    load_start=time.time()
    best_model = tf.keras.models.load_model(save_path)
    load_end=time.time()

    # --- evaluate on test set ---
    test_loss, test_acc = best_model.evaluate(x_test_norm, y_test, verbose=verbose)
    end_test_time=time.time()
    test_duration=end_test_time-end_train_time-(load_end-load_start)
    train_duration=end_train_time-start_train_time
    file_bytes = os.path.getsize(save_path)
    model_size_mb = file_bytes / (1024 * 1024)
    num_params=best_model.count_params()
    return train_duration,test_duration,model_size_mb, test_loss, test_acc,num_params

def get_dct_matrix(N):
    """Computes the NxN DCT-II transformation matrix C."""
    # Formula for DCT-II matrix
    n = np.arange(N)
    k = np.arange(N).reshape((N, 1))
    
    # Cosine basis
    matrix = np.cos((np.pi/N) * (n + 0.5) * k)
    
    # Normalization (Orthogonal DCT)
    matrix[0, :] *= 1 / np.sqrt(N)
    matrix[1:, :] *= np.sqrt(2 / N)
    
    return tf.constant(matrix, dtype=tf.float32)

# ---------------------------------------------------------
# GLOBAL SETUP (Run Once)
# ---------------------------------------------------------
# Create the 28x28 transformation matrix
DCT_MAT = get_dct_matrix(28)
DCT_MAT_T = tf.transpose(DCT_MAT) # Pre-transpose for the second mult
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

x_test=np.array(x_test,dtype=np.float64)
y_test=np.array(y_test)
x_train=np.array(x_train,dtype=np.float64)
y_train=np.array(y_train)

x_test=x_test/255
x_train=x_train/255

x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train,random_state=42)
#percent_array=[0.0001]
percent_array=(np.array([0.1,0.25,0.5,0.75,1,2.5,5,7.5,10,25,50,75,100])/100).tolist()
test_loss_arr=[]
test_acc_arr=[]
coefficients_arr=[]
psnr_arr=[]
train_dur_arr=[]
test_dur_arr=[]
model_size_arr=[]
runtime_arr=[]
model_parameter_arr=[]
percent_arr=[]
ITERATIONS=1
iterate_arr=[]
for i in range(ITERATIONS):
    print(f"=================Iteration Number {i+1}=====================")
    for percent in percent_array:
        iterate_arr.append(i)
        percent_arr.append(percent)
        print(f"-----------Percentage={percent}----------------")

        #Training Dataset Preprocessing
        train_pp_start_time=time.time()
        run_start=time.time()
        mask=create_masks(x_train,y_train,percent)
        indices_tensor=get_fixed_indices(mask)
        print(np.sum(mask))
        
        x_tr_tensor = tf.convert_to_tensor(x_tr, dtype=tf.float32)
        x_val_tensor = tf.convert_to_tensor(x_val, dtype=tf.float32)
        x_tr_in = tf_dct2d_preprocess(x_tr_tensor, indices_tensor).numpy()
        x_val_in = tf_dct2d_preprocess(x_val_tensor, indices_tensor).numpy()
        train_pp_end_time=time.time()

        #PSNR Calculation. Doesnt Contribute to time
        psnr_arr.append(compute_psnr(x_tr,mask))

        #Test DatasetPrepreocessing
        test_pp_start_time=time.time()
        x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
        x_test_in = tf_dct2d_preprocess(x_test_tensor, indices_tensor).numpy()
        test_pp_end_time=time.time()

        #Running Model
        train_dur,test_dur,model_size, test_loss, test_acc,num_params=train_and_test(x_tr_in,y_tr,x_val_in,y_val,x_test_in,y_test)\
        
        coefficients_arr.append(np.sum(mask))
        model_parameter_arr.append(num_params)
        test_loss_arr.append(test_loss)
        test_acc_arr.append(test_acc)
        train_dur_arr.append(train_dur+train_pp_end_time-train_pp_start_time)
        print("Testing Preprocessing = ",test_pp_end_time-test_pp_start_time)
        test_dur_arr.append(test_dur+test_pp_end_time-test_pp_start_time)
        model_size_arr.append(model_size)




    
print(percent_arr)
print(test_loss_arr)
print(test_acc_arr)
print(coefficients_arr)
print(psnr_arr)
print(train_dur_arr)
print(test_dur_arr)
print(model_size_arr)
df = pd.DataFrame({
    'Percent': percent_arr,
    'Test_Loss': test_loss_arr,
    'Test_Accuracy': test_acc_arr,
    'Coefficients': coefficients_arr,
    'Average PSNR': psnr_arr,
    'Train Duration':train_dur_arr,
    'Test Duration':test_dur_arr,
    'Model Size':model_size_arr,
    'Model Parameters':model_parameter_arr,
    'Iteration Number':iterate_arr
})
df.to_csv(r'C:\Users\Mayukh\Documents\KGP\Sem8\MNIST\experiment_results_dct2.csv', index=False)
print("Saved")