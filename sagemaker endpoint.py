#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Specify your S3 paths for training and validation data
train_data_s3 = 's3://prediction-tumor/brain/Training/'
valid_data_s3 = 's3://prediction-tumor/brain/Testing/'

# Create the TensorFlow Estimator
estimator = TensorFlow(
    entry_point='train.py',  # Path to the script (can be local or S3)
    role=role,
    instance_count=1,  # Use 1 instance for training
    instance_type='ml.m5.xlarge',  # Instance type (adjust as per requirement)
    output_path='s3://prediction-tumor/data/output',  # S3 path to save the model
    framework_version='2.12.0',  # TensorFlow version
    py_version='py310',  # Python version
    hyperparameters={
        'batch_size': 32,  # Hyperparameter: batch size
        'epochs': 50  # Hyperparameter: epochs
    },
    sagemaker_session=sagemaker_session
)

# Specify the inputs for the estimator, the training and validation data
inputs = {
    'train': train_data_s3,  # S3 path for the training data
    'validation': valid_data_s3  # S3 path for the validation data
}

# Start the training job
estimator.fit(inputs)


# In[2]:


import boto3

client = boto3.client('sagemaker')

response = client.describe_training_job(TrainingJobName='tensorflow-training-2024-10-09-08-02-42-019')
print(response['TrainingJobStatus'])


# In[3]:


import sagemaker
from sagemaker import get_execution_role
import boto3

# Define the S3 path where the model is stored
model_s3_path = 's3://prediction-tumor/data/output/tensorflow-training-2024-10-09-08-02-42-019/model'

# You can use the s3 client to list the contents of the model path
s3_client = boto3.client('s3')

response = s3_client.list_objects_v2(Bucket='prediction-tumor', Prefix='data/output/tensorflow-training-2024-10-09-08-02-42-019/model')
print(response)


# In[1]:


get_ipython().system('pip install pandas numpy matplotlib boto3 tensorflow scikit-learn pillow')


# In[2]:


import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
import boto3
import numpy as np
import s3fs
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
from sagemaker import image_uris

# Get the SageMaker execution role (this is the IAM role used for deployment)
role = get_execution_role()

# S3 path to the trained model (make sure this is the correct path)
model_path = 's3://prediction-tumor/data/output/tensorflow-training-2024-10-09-08-02-42-019/output/model.tar.gz'

# Check if the model path is accessible
s3_client = boto3.client('s3')
bucket_name = model_path.split('/')[2]
model_key = '/'.join(model_path.split('/')[3:])

try:
    s3_client.head_object(Bucket=bucket_name, Key=model_key)
    print(f"Model exists at {model_path}")
except Exception as e:
    print(f"Error accessing model at {model_path}: {e}")
    raise

# Get the correct TensorFlow image URI based on the TensorFlow version and region
tensorflow_image_uri = image_uris.retrieve(
    framework="tensorflow",
    region=boto3.Session().region_name,
    version="2.11.1",  # Use the latest supported version (2.11.1)
    image_scope="inference",  # Use the inference image for deployment
    instance_type="ml.m5.large"  # Explicitly specify the instance type
)

# Create the SageMaker model object
try:
    model = Model(
        model_data=model_path,
        role=role,
        image_uri=tensorflow_image_uri  # Explicitly provide the image URI for TensorFlow
    )
    print("Model object created successfully.")
except Exception as e:
    print(f"Failed to create the model object: {e}")
    raise

# Deploy the model to a SageMaker endpoint
endpoint_name = "tumor-classifier-endpoint-v9"  # Custom name for the endpoint

# Deploy the model and check for success
try:
    predictor = model.deploy(
        initial_instance_count=1,   # Number of instances to deploy
        instance_type="ml.m5.large",  # Instance type to use
        endpoint_name=endpoint_name  # Custom endpoint name
    )
    print(f"Model deployed successfully to endpoint: {endpoint_name}")

    # Check the status of the endpoint
    client = boto3.client('sagemaker')
    response = client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_status = response['EndpointStatus']
    print(f"Endpoint status: {endpoint_status}")

    # If the endpoint status is 'InService', the model is deployed successfully
    if endpoint_status == 'InService':
        print(f"Endpoint '{endpoint_name}' is up and running.")

except Exception as e:
    print(f"Failed to deploy the model: {e}")
    raise


# In[3]:


import json
import numpy as np
import boto3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io
import s3fs

# Initialize SageMaker runtime client
runtime_client = boto3.client('sagemaker-runtime')

# Define class labels and their mapping
class_indices = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
class_labels = list(class_indices.keys())

# Function to preprocess image from S3 before sending it to SageMaker
def preprocess_image_from_s3(image_path, target_size=(224, 224)):
    """Load an image from S3, preprocess it, and return a numpy array."""
    s3 = s3fs.S3FileSystem(anon=False)
    with s3.open(image_path, 'rb') as file:
        image_bytes = file.read()
        img = load_img(io.BytesIO(image_bytes), target_size=target_size)
    img_array = img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return img_array

# Function to make prediction using SageMaker Endpoint
def predict_with_endpoint(image, endpoint_name):
    """Send image to SageMaker endpoint and get predictions."""
    # Convert image to a JSON-compatible format (list of lists, batch of images)
    payload = json.dumps(image.tolist())
    
    try:
        # Call the endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',  # Change if your model expects a different format
            Body=payload
        )

        # Check the response status code
        status_code = response['ResponseMetadata']['HTTPStatusCode']
        if status_code == 200:
            # Extract the prediction result
            result = json.loads(response['Body'].read().decode())
            print("Prediction response from SageMaker:", result)  # Debugging output
            return result
        else:
            raise Exception(f"Error in prediction. Status code: {status_code}")
    
    except Exception as e:
        print(f"Failed to predict with endpoint: {e}")
        return None

# Function to extract predicted class from model's output
def extract_prediction(prediction, class_indices):
    """Extract the predicted class and its confidence from model output."""
    print(f"Extracting prediction from: {prediction}")  # Debugging output
    
    # Assuming the model outputs class probabilities (as an array)
    if isinstance(prediction, dict) and 'predictions' in prediction:
        prediction_probs = prediction['predictions'][0]  # For single image predictions
    else:
        raise ValueError("Unexpected prediction format.")
    
    print(f"Prediction probabilities: {prediction_probs}")  # Debugging output

    predicted_class_idx = np.argmax(prediction_probs)
    predicted_label = class_labels[predicted_class_idx]  # Map index to class label
    confidence = prediction_probs[predicted_class_idx]  # Confidence score for the predicted class
    return predicted_label, confidence

# Example: Predicting a single image
def predict_image(image_path, endpoint_name):
    """Load an image from S3, preprocess it, send to the SageMaker endpoint, and return prediction."""
    # Preprocess the image
    image = preprocess_image_from_s3(image_path)
    
    # Make prediction
    prediction = predict_with_endpoint(image, endpoint_name)
    
    if prediction:
        # Extract predicted label and confidence
        predicted_label, confidence = extract_prediction(prediction, class_indices)
        print(f"Predicted Label: {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
    else:
        print("Prediction failed!")

# Replace with your actual endpoint name
endpoint_name = 'tumor-classifier-endpoint-v9'  # SageMaker Endpoint name

# Provide the S3 path to your test image (update this with a valid S3 image path)
image_path = 's3://prediction-tumor/brain/Testing/glioma/Te-gl_0010.jpg'  # Path to test image

# Call the function to make the prediction
predict_image(image_path, endpoint_name)


# In[4]:


import boto3

# Initialize the SageMaker client
sagemaker_client = boto3.client('sagemaker')

# Specify the name of the endpoint to delete
endpoint_name = 'tumor-classifier-endpoint-v8'

# Delete the endpoint
try:
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"Successfully deleted endpoint: {endpoint_name}")
except sagemaker_client.exceptions.ResourceNotFound as e:
    print(f"Endpoint {endpoint_name} does not exist or has already been deleted.")
except Exception as e:
    print(f"Error deleting the endpoint: {e}")


# In[ ]:




