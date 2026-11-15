# Flask Web Application for Brain Tumor Classification
# Chapter 5.7: User Interface Design - Web Interface

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import numpy as np
import pickle
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load trained models
try:
    with open('svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('logistic_model.pkl', 'rb') as f:
        lg_model = pickle.load(f)
    models_loaded = True
except:
    models_loaded = False

# Class mappings
classes = {'no_tumor': 0, 'pituitary_tumor': 1, 'glioma_tumor': 2, 'meningioma_tumor': 3}
dec = {v: k for k, v in classes.items()}

def preprocess_image(image_path):
    """Preprocess image for prediction"""
    img = cv2.imread(image_path, 0)
    img_resized = cv2.resize(img, (200, 200))
    img_flattened = img_resized.reshape(1, -1) / 255.0
    return img_flattened, img_resized

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess image
            img_processed, img_display = preprocess_image(filepath)
            
            # Make predictions
            svm_pred = svm_model.predict(img_processed)[0]
            svm_prob = svm_model.predict_proba(img_processed).max()
            
            lg_pred = lg_model.predict(img_processed)[0]
            lg_prob = lg_model.predict_proba(img_processed).max()
            
            # Convert image to base64 for display
            plt.figure(figsize=(6, 6))
            plt.imshow(img_display, cmap='gray')
            plt.axis('off')
            plt.title('Uploaded MRI Image')
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            results = {
                'svm_prediction': dec[svm_pred].replace('_', ' ').title(),
                'svm_confidence': f"{svm_prob:.2%}",
                'lg_prediction': dec[lg_pred].replace('_', ' ').title(),
                'lg_confidence': f"{lg_prob:.2%}",
                'image': image_base64
            }
            
            return jsonify(results)
            
        except Exception as e:
            return jsonify({'error': f'Processing error: {str(e)}'})
        
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/about')
def about():
    """About page with system information"""
    system_info = {
        'models': ['SVM', 'Logistic Regression'],
        'classes': ['No Tumor', 'Pituitary Tumor', 'Glioma Tumor', 'Meningioma Tumor'],
        'features': ['Image preprocessing', 'Multi-model prediction', 'Confidence scoring'],
        'accuracy': {'SVM': '95%', 'Logistic Regression': '92%'}
    }
    return render_template('about.html', info=system_info)

@app.route('/documentation')
def documentation():
    """Documentation page"""
    return render_template('documentation.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)