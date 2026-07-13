import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Import our utilities
import utils

def preprocess_image(img_path, target_size=(224, 224)):
    """Loads and preprocesses an image for the MobileNetV2 model."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found at: {img_path}")
        
    try:
        # Load using PIL to handle different formats and verify integrity
        img = Image.open(img_path)
        img = img.convert('RGB') # Ensure it is 3-channel RGB
        img = img.resize(target_size)
        
        # Convert to numpy array and rescale
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        return x
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {e}")

def run_inference(image_path, model_path='food_model.keras', classes_path='class_names.pkl'):
    """Loads the model and classes, runs inference on the image, and returns results."""
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Trained model file '{model_path}' not found.")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
        
    if not os.path.exists(classes_path):
        print(f"Error: Class names pickle file '{classes_path}' not found.")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
        
    print(f"Loading model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    # Load class names
    classes = utils.load_pickle(classes_path)
    print(f"Loaded class categories: {classes}")
    
    # Preprocess image
    try:
        processed_image = preprocess_image(image_path)
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(1)
        
    # Run prediction
    print("Running model prediction...")
    preds = model.predict(processed_image, verbose=0)[0]
    
    # Get Top-3 indices and confidence values
    top_indices = np.argsort(preds)[::-1][:3]
    
    results = []
    for idx in top_indices:
        results.append({
            'class_name': classes[idx],
            'confidence': float(preds[idx])
        })
        
    # Print results
    top_prediction = results[0]
    predicted_class = top_prediction['class_name']
    confidence = top_prediction['confidence'] * 100
    
    print("\n=============================================")
    print("🍕 INFERENCE RESULT 🍕")
    print("=============================================")
    print(f"Predicted Class: {predicted_class.upper()}")
    print(f"Confidence:      {confidence:.2f}%")
    
    # Funny message
    funny_msg = utils.FUNNY_MESSAGES.get(predicted_class, "Yum! Looks delicious! 😋")
    print(f"Remarks:         \"{funny_msg}\"")
    print("---------------------------------------------")
    print("Top 3 Predictions:")
    for i, res in enumerate(results):
        print(f" {i+1}. {res['class_name']}: {res['confidence']*100:.2f}%")
    print("=============================================\n")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Food Category Recognition Command Line Predictor")
    parser.add_argument('image', type=str, help="Path to the input image file to classify")
    parser.add_argument('--model', type=str, default='food_model.keras', help="Path to the trained keras model file")
    parser.add_argument('--classes', type=str, default='class_names.pkl', help="Path to class names pickle file")
    
    args = parser.parse_args()
    run_inference(args.image, model_path=args.model, classes_path=args.classes)
