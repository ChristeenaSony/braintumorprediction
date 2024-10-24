import os
import pandas as pd
import boto3
import s3fs
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import io
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from keras import backend as K

# S3 bucket and dataset details
bucket_name = 'bone-cancer'
train_prefix = 'dataset/train'
test_prefix = 'dataset/test'
val_prefix = 'dataset/val'

# Image size and batch size
img_size = (224, 224)  # Resize images to (224, 224) for the model
batch_size = 32  # Define batch size for training

# Initialize the S3 client
s3 = boto3.client('s3')

# Function to list all files in a given S3 prefix (directory)
def list_files_in_s3(prefix):
    files = []
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    while response.get('Contents'):
        for item in response['Contents']:
            files.append(item['Key'])
        
        if response.get('IsTruncated'):  
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=response['NextContinuationToken'])
        else:
            break

    return files

# Prepare the image DataFrame from S3
def prepare_image_df_s3(prefix):
    labels = []
    images = []
    
    folder_files = list_files_in_s3(prefix)
    
    print(f"Files in {prefix}: {len(folder_files)} found.")
    
    for file_path in folder_files:
        label = file_path.split('/')[2]  # Assuming the label is in the folder name
        labels.append(label)
        s3_path = f"s3://{bucket_name}/{file_path}"
        images.append(s3_path)
    
    df = pd.DataFrame({
        'Img_path': images,  # Full S3 URLs
        'Img_label': labels
    })

    return df

# Prepare train, validation, and test DataFrames
train_df = prepare_image_df_s3(train_prefix)
test_df = prepare_image_df_s3(test_prefix)
val_df = prepare_image_df_s3(val_prefix)

# Ensure paths are strings
train_df['Img_path'] = train_df['Img_path'].astype(str)
val_df['Img_path'] = val_df['Img_path'].astype(str)
test_df['Img_path'] = test_df['Img_path'].astype(str)

# Ensure labels are strings
train_df['Img_label'] = train_df['Img_label'].astype(str)
val_df['Img_label'] = val_df['Img_label'].astype(str)
test_df['Img_label'] = test_df['Img_label'].astype(str)

# Custom image loader for S3 images with error handling
def load_s3_image(path, target_size):
    try:
        s3 = s3fs.S3FileSystem(anon=False)
        
        with s3.open(path, 'rb') as file:
            image_bytes = file.read()
            
            # Check if the image is valid (by reading its header)
            try:
                img = load_img(io.BytesIO(image_bytes), target_size=target_size)
            except UnidentifiedImageError as e:
                print(f"Failed to load image {path}: {e}")
                return None  # Return None for invalid images
            return img_to_array(img)
    except Exception as e:
        print(f"Failed to read image {path}: {e}")
        return None

# Custom generator to load images from S3 for the ImageDataGenerator
def s3_image_generator(df, batch_size, img_size, class_indices, shuffle=True):
    num_samples = len(df)
    
    while True:
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data at the start of each epoch
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_paths = df['Img_path'].iloc[start:end].values
            batch_labels = df['Img_label'].iloc[start:end].values

            images = []
            for path in batch_paths:
                img = load_s3_image(path, img_size)
                if img is not None:
                    images.append(img)
            
            # Skip empty batches if no valid images found
            if len(images) == 0:
                continue

            labels = np.array([class_indices.get(label, -1) for label in batch_labels])
            
            if -1 in labels:  # Skip batches with unknown labels
                continue

            # Debug: Check batch size
            if len(images) != len(labels):
                print(f"Mismatch in batch size! Images: {len(images)}, Labels: {len(labels)}")

            # Ensure the labels are integers (not one-hot encoded for binary classification)
            labels = labels.reshape(-1, 1)  # Reshape labels to (batch_size, 1)

            yield np.array(images), labels

# Get class indices
class_indices = {label: idx for idx, label in enumerate(train_df['Img_label'].unique())}
class_count = len(class_indices)  # Number of classes

# Steps per epoch and validation steps
steps_per_epoch = len(train_df) // batch_size
validation_steps = len(val_df) // batch_size
test_steps = len(test_df) // batch_size

# Create custom generators with shuffle for train and valid generators
train_generator = s3_image_generator(train_df, batch_size=batch_size, img_size=img_size, class_indices=class_indices, shuffle=True)
valid_generator = s3_image_generator(val_df, batch_size=batch_size, img_size=img_size, class_indices=class_indices, shuffle=True)

# Test generator (without shuffling)
test_generator = s3_image_generator(test_df, batch_size=batch_size, img_size=img_size, class_indices=class_indices, shuffle=False)

# Model definition
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))  # Binary classification with 1 output neuron

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

# Define callbacks
checkpoint_cb = ModelCheckpoint("/tmp/Best_model.keras", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)

# Train the model
hist = model.fit(
    train_generator,
    epochs=10,
    validation_data=valid_generator,
    callbacks=[checkpoint_cb, early_stopping_cb],
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

# Save the model
model_save_path = '/opt/ml/model/2'
os.makedirs(model_save_path, exist_ok=True)
model.save(model_save_path)

# Evaluate the model on the test data
test_loss, test_acc, test_auc = model.evaluate(test_generator, verbose=2)

# Print test results
print(f'\nTest Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')
print(f'Test AUC: {test_auc}')

# Add completion message
print("Training job completed successfully. Model saved at:", model_save_path)
