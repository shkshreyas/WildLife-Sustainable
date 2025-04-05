from flask import Flask, render_template, request, jsonify, Response, url_for
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os
import base64
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/wildlife_classifier_model.pth'  # Path to your saved model
CLASS_NAMES_PATH = 'model/class_names.json'  # Path to your class names JSON file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if model exists, otherwise provide instructions
model_exists = os.path.exists(MODEL_PATH)

# Load the model if it exists
if model_exists:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()
    
    # Load class names
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
else:
    model = None
    class_names = ["cane", "cavallo", "elefante", "farfalla", "gallina", "gatto", "mucca", "pecora", "ragno", "scoiattolo"]

# Define image transformation - same as used in training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image_and_predict(img):
    """
    Process image and make prediction - shared logic between endpoints
    """
    if not model_exists:
        return {'error': 'Model not found. Please train and export your model first.'}, 503
        
    try:
        # Process the image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            
            # Get probabilities
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            top_5_prob, top_5_indices = torch.topk(probabilities, 5)
            
            results = {
                'prediction': predicted_class,
                'confidence': float(probabilities[predicted].item()),
                'top_5': [
                    {'class': class_names[idx.item()], 
                     'probability': float(prob.item())}
                    for prob, idx in zip(top_5_prob, top_5_indices)
                ]
            }
            
        return results, 200
    except Exception as e:
        return {'error': f'Error during prediction: {str(e)}'}, 500

# Web interface routes
@app.route('/')
def index():
    return render_template('index.html', model_available=model_exists)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload an image (png, jpg, jpeg, gif)'}), 400
    
    try:
        # Process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        results, status_code = process_image_and_predict(img)
        return jsonify(results), status_code
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/export_guide')
def export_guide():
    return render_template('export_guide.html')

# API Routes
@app.route('/api', methods=['GET'])
def api_documentation():
    """API documentation endpoint"""
    # Check if the request prefers HTML (browser) or JSON (API client)
    if request.headers.get('Accept', '').find('text/html') != -1:
        # Browser request - render HTML template
        return render_template('api_docs.html', url_root=request.url_root)
    else:
        # API client request - return JSON
        base_url = request.url_root
        docs = {
            'name': 'Wildlife Classification API',
            'version': '1.0',
            'description': 'API for classifying wildlife images',
            'endpoints': [
                {
                    'path': '/api/predict',
                    'method': 'POST',
                    'description': 'Classify a wildlife image',
                    'parameters': [
                        {
                            'name': 'file',
                            'type': 'file',
                            'description': 'Image file to classify (png, jpg, jpeg, gif)'
                        }
                    ],
                    'example': f'curl -X POST -F "file=@your_image.jpg" {base_url}api/predict'
                },
                {
                    'path': '/api/predict/url',
                    'method': 'POST',
                    'description': 'Classify a wildlife image from a URL',
                    'parameters': [
                        {
                            'name': 'image_url',
                            'type': 'string',
                            'description': 'URL of the image to classify'
                        }
                    ],
                    'example': f'curl -X POST -H "Content-Type: application/json" -d {{"image_url": "https://example.com/image.jpg"}} {base_url}api/predict/url'
                },
                {
                    'path': '/api/predict/base64',
                    'method': 'POST',
                    'description': 'Classify a wildlife image from base64 encoded string',
                    'parameters': [
                        {
                            'name': 'image_data',
                            'type': 'string',
                            'description': 'Base64 encoded image data'
                        }
                    ],
                    'example': f'curl -X POST -H "Content-Type: application/json" -d {{"image_data": "base64_encoded_image_data"}} {base_url}api/predict/base64'
                }
            ]
        }
        return jsonify(docs)

@app.route('/api/predict', methods=['POST'])
def api_predict_file():
    """API endpoint for predicting from uploaded file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload an image (png, jpg, jpeg, gif)'}), 400
    
    try:
        # Process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        results, status_code = process_image_and_predict(img)
        return jsonify(results), status_code
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/api/predict/url', methods=['POST'])
def api_predict_url():
    """API endpoint for predicting from image URL"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({'error': 'No image_url provided in request'}), 400
    
    try:
        # Download the image
        response = requests.get(data['image_url'], stream=True)
        response.raise_for_status()
        
        # Process the image
        img = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        results, status_code = process_image_and_predict(img)
        return jsonify(results), status_code
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Error downloading image: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

@app.route('/api/predict/base64', methods=['POST'])
def api_predict_base64():
    """API endpoint for predicting from base64 encoded image"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if 'image_data' not in data:
        return jsonify({'error': 'No image_data provided in request'}), 400
    
    try:
        # Decode base64 image
        image_data = data['image_data']
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
            
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        results, status_code = process_image_and_predict(img)
        return jsonify(results), status_code
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)