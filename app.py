"""
app.py
Flask web application for AI Image Authenticity Detection.
"""

import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import timm
from torchvision import transforms
import numpy as np

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
CLASS_LABELS = ['AI Generated', 'Real']

# Global model variable
model = None

def load_model():
    """Load the trained ViT model."""
    global model
    
    if model is None:
        print("Loading model...")
        
        # Create model architecture
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        model.head = nn.Linear(model.head.in_features, 2)
        
        # Load trained weights
        model_path = 'models/model_vit.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        print("Model loaded successfully!")
    
    return model

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """
    Predict whether an image is AI-generated or real.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        label: Predicted class label
        confidence: Confidence score (percentage)
    """
    # Load model
    model = load_model()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item() * 100
    
    return CLASS_LABELS[predicted_class], confidence_score

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    # Check if file was uploaded
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    
    # Check if file was selected
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return render_template('index.html', error='Invalid file type. Allowed types: png, jpg, jpeg, gif, bmp, webp')
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        label, confidence = predict_image(filepath)
        
        # Prepare result
        result = {
            'label': label,
            'confidence': f'{confidence:.2f}',
            'image_url': url_for('static', filename=f'uploads/{filename}')
        }
        
        return render_template('index.html', result=result)
    
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {str(e)}')

@app.route('/about')
def about():
    """About page with model information."""
    return render_template('index.html', show_about=True)

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists('models/model_vit.pth'):
        print("=" * 60)
        print("WARNING: Model file not found!")
        print("Please run 'python train.py' to train the model first.")
        print("=" * 60)
    
    # Run Flask app
    print("\n" + "=" * 60)
    print("Starting AI Image Authenticity Detector")
    print("=" * 60)
    print(f"Device: {device}")
    print("Open your browser and navigate to: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)