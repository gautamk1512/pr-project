# Pattern Recognition Project

This repository contains a collection of pattern recognition and machine learning implementations, along with a web-based UI component for demonstration purposes.

## Project Components

### Web UI Component
- Modern carousel interface built with HTML, CSS, and JavaScript
- Responsive design with smooth animations
- Team profile display functionality

### Machine Learning Components

#### SVM Classification
- Implementation of Support Vector Machine (SVM) with different kernels
- Comparison with other classification algorithms (Logistic Regression, Decision Tree, Random Forest, KNN)
- Performance evaluation using accuracy metrics

#### Face Detection
- Real-time face and eye detection using OpenCV
- Implementation using Haar Cascade Classifiers
- Video capture and processing capabilities

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.7+
- Required Python packages (see `requirements.txt`)
- Web browser for UI components

### Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Running the Applications

#### Web UI
Open `index.html` in a web browser to view the carousel component.

#### SVM Classification
Run the SVM comparison script:
```
python main.py
```

#### Face Detection
Run the face detection script:
```
python myfacedetection.py
```
Note: This requires a webcam and the Haar cascade XML files in the project directory.

## Project Structure

- `index.html`, `styles.css`, `script.js`: Web UI carousel component
- `main.py`: SVM classification comparison
- `myfacedetection.py`: Face detection implementation
- `haarcascade_frontalcatface.xml`, `haarcascade_eye.xml`: Cascade files for face detection

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for computer vision capabilities
- scikit-learn for machine learning implementations
- Unsplash for profile images used in the carousel