from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import json
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = 'model/wildlife_classifier_model.pth'  # Path to your saved model
CLASS_NAMES_PATH = 'model/class_names.json'  # Path to your class names JSON file

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

@app.route('/')
def index():
    return render_template('index.html', model_available=model_exists)

@app.route('/predict', methods=['POST'])
def predict():
    if not model_exists:
        return jsonify({
            'error': 'Model not found. Please train and export your model first.'
        })
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    try:
        # Process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
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
            
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'})

@app.route('/export_guide')
def export_guide():
    return render_template('export_guide.html')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True)