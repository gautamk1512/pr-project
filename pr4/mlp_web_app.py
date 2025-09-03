# Multi-Layer Perceptron (MLP) Web Application
# Dataset: MNIST Handwritten Digits

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, jsonify
import os
import json

app = Flask(__name__)

# Global variables
model = None
x_test = None
y_test = None
history_data = None

def train_and_save_model():
    """Train the MLP model and save it"""
    global history_data
    
    # 1. Load dataset
    (x_train, y_train), (x_test_global, y_test_global) = mnist.load_data()
    
    # Normalize data (0-255 → 0-1)
    x_train = x_train / 255.0
    x_test_global = x_test_global / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test_global = to_categorical(y_test_global, 10)
    
    # 2. Build MLP model
    model = Sequential([
        Flatten(input_shape=(28, 28)),   # Flatten 28x28 → 784
        Dense(128, activation='relu'),   # Hidden Layer 1
        Dense(64, activation='relu'),    # Hidden Layer 2
        Dense(10, activation='softmax')  # Output Layer (10 classes)
    ])
    
    # 3. Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 4. Train model
    history = model.fit(x_train, y_train,
                        validation_data=(x_test_global, y_test_global),
                        epochs=5,
                        batch_size=128,
                        verbose=1)
    
    # Save model
    model.save('mlp_model.h5')
    
    # Save history data
    history_data = {
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy'],
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    }
    
    with open('training_history.json', 'w') as f:
        json.dump(history_data, f)
    
    return model, x_test_global, y_test_global

def load_model_and_data():
    """Load the trained model and test data"""
    global model, x_test, y_test, history_data
    
    if os.path.exists('mlp_model.h5'):
        model = load_model('mlp_model.h5')
        print("Model loaded successfully!")
    else:
        print("Training new model...")
        model, x_test, y_test = train_and_save_model()
        return
    
    # Load test data
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test / 255.0
    y_test = to_categorical(y_test, 10)
    
    # Load history data
    if os.path.exists('training_history.json'):
        with open('training_history.json', 'r') as f:
            history_data = json.load(f)

def predict_digit(digit_index):
    """Predict a digit from the test set"""
    if model is None or x_test is None:
        return None, None, None
    
    # Get the sample
    sample = x_test[digit_index].reshape(1, 28, 28)
    actual_digit = np.argmax(y_test[digit_index])
    
    # Make prediction
    prediction = model.predict(sample, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Generate image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(x_test[digit_index], cmap='gray')
    ax.set_title(f'Actual: {actual_digit}, Predicted: {predicted_digit}\nConfidence: {confidence:.2f}%')
    ax.axis('off')
    
    # Convert plot to base64 string
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_string = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)
    
    return predicted_digit, confidence, img_string, actual_digit

def get_model_accuracy():
    """Get overall model accuracy"""
    if model is None or x_test is None or y_test is None:
        return None
    
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy * 100

def analyze_digit_pattern(predicted_digit, actual_digit, confidence):
    """Analyze the pattern and characteristics of the digit prediction"""
    pattern_info = {
        'is_correct': bool(predicted_digit == actual_digit),
        'confidence_level': 'High' if confidence > 90 else 'Medium' if confidence > 70 else 'Low',
        'digit_characteristics': get_digit_characteristics(predicted_digit),
        'prediction_quality': 'Excellent' if confidence > 95 else 'Good' if confidence > 85 else 'Fair' if confidence > 70 else 'Poor'
    }
    
    # Add specific pattern analysis for digits 0-9
    if predicted_digit == actual_digit:
        pattern_info['match_status'] = 'Perfect Match'
    else:
        pattern_info['match_status'] = f'Mismatch: Predicted {predicted_digit}, Actual {actual_digit}'
        pattern_info['common_confusion'] = get_common_confusion_pairs(predicted_digit, actual_digit)
    
    return pattern_info

def get_digit_characteristics(digit):
    """Get characteristics of each digit for pattern analysis"""
    characteristics = {
        0: 'Circular/oval shape, closed loop',
        1: 'Vertical line, simple structure',
        2: 'Curved top, horizontal bottom',
        3: 'Two curved sections, open right',
        4: 'Vertical and horizontal lines, open bottom',
        5: 'Horizontal top, curved bottom',
        6: 'Curved with closed bottom loop',
        7: 'Horizontal top, diagonal line',
        8: 'Two closed loops, figure-eight',
        9: 'Closed top loop, curved bottom'
    }
    return characteristics.get(digit, 'Unknown digit')

def get_common_confusion_pairs(predicted, actual):
    """Identify common digit confusion patterns"""
    confusion_pairs = {
        (0, 6): 'Circular shapes often confused',
        (6, 0): 'Circular shapes often confused',
        (1, 7): 'Linear elements similarity',
        (7, 1): 'Linear elements similarity',
        (2, 5): 'Curved and horizontal elements',
        (5, 2): 'Curved and horizontal elements',
        (3, 8): 'Multiple curved sections',
        (8, 3): 'Multiple curved sections',
        (4, 9): 'Vertical line confusion',
        (9, 4): 'Vertical line confusion'
    }
    
    pair_key = (predicted, actual)
    return confusion_pairs.get(pair_key, f'Uncommon confusion between {predicted} and {actual}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        digit_index = int(request.json['digit_index'])
        
        # Ensure index is within valid range (0-9999 for MNIST test set)
        if digit_index < 0:
            digit_index = 0
        elif digit_index >= len(x_test):
            digit_index = digit_index % len(x_test)
        
        # For inputs 1-1000, ensure proper mapping to test set indices
        if 1 <= digit_index <= 1000:
            # Map 1-1000 to valid test indices, ensuring pattern consistency
            digit_index = (digit_index - 1) % len(x_test)
        
        predicted_digit, confidence, img_string, actual_digit = predict_digit(digit_index)
        overall_accuracy = get_model_accuracy()
        
        # Add pattern analysis for the predicted digit
        pattern_info = analyze_digit_pattern(predicted_digit, actual_digit, confidence)
        
        return jsonify({
            'predicted_digit': int(predicted_digit),
            'actual_digit': int(actual_digit),
            'confidence': float(confidence),
            'overall_accuracy': float(overall_accuracy),
            'image': img_string,
            'history': history_data,
            'pattern_info': pattern_info,
            'input_index': digit_index
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("Loading model and data...")
    load_model_and_data()
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)