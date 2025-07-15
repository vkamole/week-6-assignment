# ♻️ AI Waste Classifier - Edge Deployment

An lightweight TensorFlow Lite model that classifies waste as **recyclable** or **organic**, optimized for Raspberry Pi.

## Features
- 92.5% test accuracy
- 9.07MB TFLite model
- Real-time inference (15 FPS on RPi 4)

## Quick Start
### Requirements
- Python 3.7+
- TensorFlow 2.x or `tflite-runtime`

### Installation
```bash
pip install -r requirements.txt  # For full TensorFlow
# OR for Raspberry Pi:
pip install tflite-runtime pillow numpy
Usage
python
from classifier import WasteClassifier

clf = WasteClassifier('waste_classifier.tflite')
result = clf.predict('test.jpg')
print(result)  # {'class': 'recyclable', 'confidence': 0.92}
Run on Raspberry Pi
bash
python3 classify.py --image_path test.jpg
# Output: {"prediction": "recyclable", "score": 0.87}
File Structure
text
.
├── waste_classifier.tflite      # Deployable model
──  training_notebook.ipynb      # Model development
└── test_images/                 # Sample images